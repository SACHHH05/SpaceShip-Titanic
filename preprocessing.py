import pandas as pd

RANDOM_SEED = 42
DATASET_DIR = "dataset"
TRAIN_PATH = f"{DATASET_DIR}/train.csv"
TEST_PATH = f"{DATASET_DIR  }/test.csv"

NUMERIC_COLS = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]

def fit_imputation_stats(train_df: pd.DataFrame) -> dict:
    stats = {"numeric_medians": {}, "categorical_modes": {}}
    for col in NUMERIC_COLS:
        stats["numeric_medians"][col] = train_df[col].median()
    for col in CATEGORICAL_COLS:
        stats["categorical_modes"][col] = train_df[col].mode(dropna=True).iloc[0]
    return stats

def apply_numeric_categorical_imputation(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(stats["numeric_medians"][col])
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(stats["categorical_modes"][col])
    return df

def impute_cabin_and_passenger_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["PassengerId"])
    df["_Group"] = df["PassengerId"].str.split("_").str[0]
    df["Cabin"] = (
        df.groupby("_Group")["Cabin"]
        .transform(lambda s: s.ffill().bfill())
    )
    df["Cabin"] = df["Cabin"].fillna("Unknown/0/Unknown")
    df = df.drop(columns=["_Group"])
    return df

def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = impute_cabin_and_passenger_group(train_df)
    test_df = impute_cabin_and_passenger_group(test_df)
    stats = fit_imputation_stats(train_df)
    train_clean = apply_numeric_categorical_imputation(train_df, stats)
    test_clean = apply_numeric_categorical_imputation(test_df, stats)
    return train_clean, test_clean, stats

if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)