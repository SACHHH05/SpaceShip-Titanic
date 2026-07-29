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