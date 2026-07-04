"""
preprocess_housets.py — HouseTS Data Preprocessing Pipeline
=============================================================
Làm sạch, loại bỏ leakage, engineer features, encode, scale bộ dữ liệu
HouseTS (panel data theo tháng x zipcode, 30 metro area, 2012-2023)
để sẵn sàng cho model training.

Cùng phong cách với California Housing pipeline: tách load_and_clean()
(dùng cho EDA, không encode/scale/split) và encode_and_scale()
(dùng riêng ở bước training, fit CHỈ trên train để tránh leakage).

Điểm khác biệt quan trọng so với California Housing:
- Đây là dữ liệu PANEL (time series x zipcode) → PHẢI split theo thời
  gian (time-based split), KHÔNG được random split, nếu không model sẽ
  "nhìn thấy tương lai" của chính zipcode đó trong tập train.
- zipcode có cardinality rất cao (6226) → không one-hot được, dùng
  target encoding (fit trên train, transform trên test).

Usage (EDA):
    from preprocess_housets import load_and_clean
    df_clean = load_and_clean("data/HouseTS.csv")

Usage (Training):
    from preprocess_housets import load_and_clean, time_based_split, encode_and_scale
    df_clean = load_and_clean("data/HouseTS.csv")
    train_df, test_df = time_based_split(df_clean, cutoff_date="2022-06-30")
    X_train, X_test, y_train, y_test, artifacts = encode_and_scale(train_df, test_df)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = "data/HouseTS.csv"

# Target column. `price` được chọn thay vì `median_sale_price` vì đây là
# bản giá đã được chuẩn hoá/impute sẵn trong dataset gốc, ít giá trị lạ hơn.
TARGET = "price"

# --- Các cột LEAKAGE: loại bỏ hoàn toàn khỏi tập feature ---
# Lý do loại từng cột:
#   median_sale_price : corr=0.92 với target, gần như là chính target
#                        (đơn vị lớn hơn nhưng cùng khái niệm "giá bán").
#   median_ppsf        : = giá bán / diện tích → công thức chứa sale price
#                        trực tiếp, chỉ cần biết diện tích là suy ra được price.
#   median_list_ppsf    : tương tự, tính từ list price / sqft, corr=0.59
#                        (list price rất gần sale price nên vẫn rò rỉ).
#   avg_sale_to_list    : = sale_price / list_price → mẫu số chứa target.
#   sold_above_list     : cờ nhị phân được tính trực tiếp từ việc so sánh
#                        sale price với list price → chứa thông tin target.
#   median_list_price   : không trực tiếp = target nhưng là giá rao của
#                        CHÍNH giao dịch đó trong nhiều trường hợp, corr
#                        thấp hơn (0.20) nên rủi ro thấp hơn nhưng vẫn giữ
#                        cẩn thận trong nhóm cảnh báo (xem LIST_PRICE_KEEP).
#   Median Home Value   : biến ước tính giá trị nhà từ Census (ACS),
#                        corr=0.83 với target — về bản chất là một phiên
#                        bản khác của "giá nhà khu vực", nếu giữ lại thì
#                        model sẽ học chủ yếu từ biến này thay vì học
#                        pattern thị trường thật.
LEAKY_COLS = [
    "median_sale_price",
    "median_ppsf",
    "median_list_ppsf",
    "avg_sale_to_list",
    "sold_above_list",
    "median_list_price",
    "Median Home Value",
]

# --- Các cột KHÔNG có tác dụng dự đoán / dư thừa ---
# city_full : trùng lặp 1-1 với city (chỉ là tên đầy đủ) → giữ city, drop city_full.
USELESS_COLS = ["city_full"]

# Cột định danh cardinality cao, không dùng trực tiếp làm feature
# (sẽ được target-encode riêng ở bước encode_and_scale, KHÔNG one-hot).
HIGH_CARDINALITY_COLS = ["zipcode"]

# Nhóm feature theo domain — tiện cho việc chọn subset khi thử nghiệm model
FEATURE_GROUPS = {
    "market_activity": ["homes_sold", "pending_sales", "new_listings",
                         "inventory", "median_dom", "off_market_in_two_weeks"],
    "poi": ["bank", "bus", "hospital", "mall", "park",
            "restaurant", "school", "station", "supermarket"],
    "demographic": ["Total Population", "Median Age", "Per Capita Income",
                     "Total Families Below Poverty", "Total Housing Units",
                     "Median Rent", "Total Labor Force", "Unemployed Population",
                     "Total School Age Population", "Total School Enrollment",
                     "Median Commute Time"],
    "geo": ["city", "zipcode"],
    "time": ["date", "year"],
}

RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Đọc CSV thô và kiểm tra sơ bộ."""
    df = pd.read_csv(path)
    print(f"[load]  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    assert TARGET in df.columns, f"Target column '{TARGET}' not found."
    assert "date" in df.columns, "Missing 'date' column."
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý missing values, giá trị bất thường, và loại leakage/cột thừa.

    Ghi chú:
    - Dataset gốc KHÔNG có missing value ở bất kỳ cột nào (đã kiểm tra),
      nhưng vẫn giữ bước impute phòng trường hợp version data khác có NaN.
    - price/median_sale_price = 0 là giá trị không hợp lệ (không có giao
      dịch thật) → loại bỏ.
    """
    df = df.copy()

    # --- 3a. Loại các dòng target không hợp lệ ---
    before = len(df)
    df = df[df[TARGET] > 0]
    print(f"[clean] Removed {before - len(df):,} rows with {TARGET} <= 0")

    # --- 3b. Impute phòng hờ (median theo city, robust hơn mean) ---
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    n_missing_before = df[numeric_cols].isnull().sum().sum()
    if n_missing_before > 0:
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df.groupby("city")[col].transform(
                    lambda x: x.fillna(x.median())
                )
                df[col] = df[col].fillna(df[col].median())  # fallback toàn cục
    print(f"[clean] Missing values before: {n_missing_before}, "
          f"after: {df[numeric_cols].isnull().sum().sum()}")

    # --- 3c. Loại cột leakage ---
    drop_leaky = [c for c in LEAKY_COLS if c in df.columns]
    df = df.drop(columns=drop_leaky)
    print(f"[clean] Dropped {len(drop_leaky)} leakage columns: {drop_leaky}")

    # --- 3d. Loại cột dư thừa/không tác dụng ---
    drop_useless = [c for c in USELESS_COLS if c in df.columns]
    df = df.drop(columns=drop_useless)
    print(f"[clean] Dropped {len(drop_useless)} redundant columns: {drop_useless}")

    print(f"[clean] Remaining rows: {len(df):,}, columns: {df.shape[1]}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo feature có ý nghĩa domain từ 4 nhóm: thời gian, tiện ích, thị
    trường, nhân khẩu học. Ưu tiên tỷ lệ (ratio/rate) hơn raw count vì
    raw count phụ thuộc quy mô dân số của từng zipcode, gây nhiễu khi
    so sánh giữa các khu vực.
    """
    df = df.copy()

    # --- 4a. Time features từ `date` ---
    # Giữ lại year (đã có sẵn), thêm month + quarter để bắt seasonality
    # (thị trường nhà thường sôi động hơn vào mùa xuân/hè).
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # --- 4b. POI: gộp thành tổng mật độ tiện ích + per-capita ---
    poi_cols = FEATURE_GROUPS["poi"]
    df["poi_total"] = df[poi_cols].sum(axis=1)
    # per 1000 dân, tránh chia 0
    df["poi_per_1000_pop"] = df["poi_total"] / (df["Total Population"] / 1000 + 1)

    # --- 4c. Nhân khẩu học: chuyển raw count → rate (giảm phụ thuộc quy mô) ---
    df["poverty_rate"] = df["Total Families Below Poverty"] / (df["Total Population"] + 1)
    df["unemployment_rate"] = df["Unemployed Population"] / (df["Total Labor Force"] + 1)
    df["school_enrollment_rate"] = df["Total School Enrollment"] / (df["Total School Age Population"] + 1)
    df["housing_units_per_capita"] = df["Total Housing Units"] / (df["Total Population"] + 1)
    # tỷ lệ tiền thuê / thu nhập (affordability proxy) — dùng thu nhập năm quy đổi
    df["rent_to_income_ratio"] = (df["Median Rent"] * 12) / (df["Per Capita Income"] + 1)

    # --- 4d. Log-transform các cột lệch phải mạnh ---
    # Total Population, Per Capita Income, inventory, homes_sold... đều
    # có phân phối lệch phải rõ rệt (một vài zipcode rất đông dân/giàu).
    skewed_cols = ["Total Population", "Per Capita Income", "inventory",
                   "homes_sold", "new_listings", "pending_sales"]
    for col in skewed_cols:
        if col in df.columns:
            df[f"log_{col.replace(' ', '_')}"] = np.log1p(df[col])

    # --- 4e. Thị trường: median_dom và off_market_in_two_weeks giữ nguyên,
    # đây là các chỉ số "tốc độ thị trường" hợp lệ, KHÔNG chứa giá trực tiếp ---
    # (không cần biến đổi thêm, chỉ ghi chú lại lý do giữ)

    print(f"[engineer] Feature count: {df.shape[1] - 1} features (target excluded)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. TARGET TRANSFORMATION
# ─────────────────────────────────────────────────────────────────────────────

def transform_target(y: pd.Series) -> pd.Series:
    """
    Log1p-transform target: price có range 10K → 8.4M, lệch phải rất mạnh.
    Khi evaluate model: dùng np.expm1() để convert ngược lại USD.
    """
    return np.log1p(y)


# ─────────────────────────────────────────────────────────────────────────────
# 6. TIME-BASED SPLIT (bắt buộc với panel data để tránh leakage thời gian)
# ─────────────────────────────────────────────────────────────────────────────

def time_based_split(df: pd.DataFrame, cutoff_date: str = "2022-06-30"):
    """
    Chia train/test theo thời gian thay vì random split.

    Lý do bắt buộc: đây là panel data (cùng 1 zipcode xuất hiện nhiều
    tháng liên tiếp). Nếu random split, các tháng liền kề của CÙNG một
    zipcode có thể vừa nằm trong train vừa nằm trong test → model "nhìn
    thấy tương lai gần" của chính điểm dữ liệu nó cần dự đoán, đánh giá
    performance sẽ bị thổi phồng giả tạo (leakage qua thời gian).

    train: date <= cutoff_date | test: date > cutoff_date
    """
    train_df = df[df["date"] <= cutoff_date].reset_index(drop=True)
    test_df = df[df["date"] > cutoff_date].reset_index(drop=True)
    print(f"[split] Train: {len(train_df):,} rows (<= {cutoff_date}) | "
          f"Test: {len(test_df):,} rows (> {cutoff_date})")
    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENCODING & SCALING
# ─────────────────────────────────────────────────────────────────────────────

def encode_and_scale(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Encode categorical + scale numeric. Tất cả encoder đều fit CHỈ trên
    train_df rồi transform sang test_df, để tránh leakage thông tin từ
    test vào quá trình encode.

    - city (30 giá trị): one-hot, cardinality thấp nên an toàn.
    - zipcode (6226 giá trị): target encoding (mean price theo zipcode,
      tính TRÊN TRAIN), vì one-hot sẽ tạo quá nhiều cột thưa. Zipcode
      không thấy trong train sẽ được gán giá trị trung bình toàn cục
      (global mean) của train.
    - date: drop khỏi feature set sau khi đã trích month/quarter/year
      (giữ lại date thô sẽ không dùng được cho model).
    - numeric còn lại: StandardScaler fit trên train.

    Returns:
        X_train, X_test, y_train, y_test, artifacts (dict chứa scaler,
        zip_target_map, global_mean, one-hot columns, feature_names)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    y_train_raw = train_df[TARGET].copy()
    y_test_raw = test_df[TARGET].copy()
    y_train = transform_target(y_train_raw)
    y_test = transform_target(y_test_raw)

    X_train = train_df.drop(columns=[TARGET])
    X_test = test_df.drop(columns=[TARGET])

    # --- 7a. Target encoding cho zipcode (fit trên train) ---
    global_mean = y_train.mean()
    zip_target_map = X_train.assign(_y=y_train).groupby("zipcode")["_y"].mean()

    X_train["zipcode_target_enc"] = X_train["zipcode"].map(zip_target_map)
    X_test["zipcode_target_enc"] = X_test["zipcode"].map(zip_target_map).fillna(global_mean)

    X_train = X_train.drop(columns=["zipcode"])
    X_test = X_test.drop(columns=["zipcode"])

    # --- 7b. One-hot city ---
    X_train = pd.get_dummies(X_train, columns=["city"], prefix="city")
    X_test = pd.get_dummies(X_test, columns=["city"], prefix="city")
    # đồng bộ cột giữa train/test (phòng city nào chỉ xuất hiện ở 1 tập)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # --- 7c. Drop cột date thô (đã trích xuất month/quarter/year) ---
    X_train = X_train.drop(columns=["date"])
    X_test = X_test.drop(columns=["date"])

    # --- 7d. StandardScale numeric columns ---
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    print(f"[encode] Target-encoded: zipcode ({len(zip_target_map)} groups)")
    print(f"[encode] One-hot encoded: city ({X_train.filter(like='city_').shape[1]} columns)")
    print(f"[scale]  StandardScaled {len(num_cols)} numeric columns")
    print(f"[encode] Final feature count: {X_train.shape[1]}")

    artifacts = {
        "scaler": scaler,
        "zip_target_map": zip_target_map,
        "global_mean": global_mean,
        "feature_names": X_train.columns.tolist(),
        "y_raw_train": y_train_raw,
        "y_raw_test": y_test_raw,
    }
    return X_train, X_test, y_train, y_test, artifacts


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN ENTRY POINT (dùng cho EDA)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Pipeline cho EDA: Load → Clean (loại leakage/cột thừa) → Engineer features.
    KHÔNG encode, KHÔNG scale, KHÔNG split.

    Returns
    -------
    df_clean : pd.DataFrame
        Gồm cả TARGET column (price, chưa log-transform), date thô, city,
        zipcode và tất cả engineered features ở dạng human-readable —
        sẵn sàng cho EDA notebook.
    """
    df = load_raw(path)
    df = clean(df)
    df = engineer_features(df)
    print(f"\n[pipeline] EDA-ready DataFrame: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI — tiện cho test nhanh
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess HouseTS Data")
    parser.add_argument("--data", default=DATA_PATH, help="Path to CSV file")
    parser.add_argument("--cutoff", default="2022-06-30",
                         help="Ngày cắt cho time-based split (train <= cutoff)")
    parser.add_argument("--output", default=None, help="Optional: thư mục lưu output parquet")
    args = parser.parse_args()

    # Bước 1: EDA-ready df
    df_clean = load_and_clean(args.data)

    # Bước 2: Time-based split (KHÔNG random split vì đây là panel data)
    train_df, test_df = time_based_split(df_clean, cutoff_date=args.cutoff)

    # Bước 3: Encode + Scale (fit trên train, transform test)
    X_train, X_test, y_train, y_test, artifacts = encode_and_scale(train_df, test_df)

    print(f"\n[pipeline] Train: {len(X_train):,} | Test: {len(X_test):,} | "
          f"Features: {X_train.shape[1]}")

    # Bước 4: Lưu nếu có output_dir
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        X_train.to_parquet(os.path.join(args.output, "X_train.parquet"), index=False)
        X_test.to_parquet(os.path.join(args.output, "X_test.parquet"), index=False)
        y_train.to_frame("price_log").to_parquet(os.path.join(args.output, "y_train.parquet"), index=False)
        y_test.to_frame("price_log").to_parquet(os.path.join(args.output, "y_test.parquet"), index=False)

        with open(os.path.join(args.output, "preprocessor_info.pkl"), "wb") as f:
            pickle.dump(artifacts, f)
        print(f"\n[save] Saved to '{args.output}/'")

    print("\n✅ Preprocessing complete.")
