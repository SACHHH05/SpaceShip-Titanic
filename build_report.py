import pandas as pd

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

def deconstruct_cabin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parts = df["Cabin"].str.split("/", expand=True)
    df["Cabin_Deck"] = parts[0]
    df["Cabin_Number"] = pd.to_numeric(parts[1], errors="coerce").fillna(-1).astype(int)
    df["Cabin_Side"] = parts[2]
    df = df.drop(columns=["Cabin"])
    return df

def extract_group_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Group"] = df["PassengerId"].str.split("_").str[0]
    df["Group_Size"] = df.groupby("Group")["PassengerId"].transform("count").astype(int)
    df["Is_Alone"] = (df["Group_Size"] == 1)
    df = df.drop(columns=["Group"])
    return df

def financial_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Total_Spend"] = df[SPEND_COLS].sum(axis=1)
    df["Zero_Spend"] = (df["Total_Spend"] == 0)
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)

    for col in ["CryoSleep", "VIP"]:
        if col in df.columns and df[col].dtype != int:
            df[col] = df[col].astype(bool).astype(int)

    text_cols = ["HomePlanet", "Destination", "Cabin_Deck", "Cabin_Side"]
    df = pd.get_dummies(df, columns=[c for c in text_cols if c in df.columns])

    new_bool_cols = df.select_dtypes(include="bool").columns.tolist()
    for col in new_bool_cols:
        df[col] = df[col].astype(int)

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = deconstruct_cabin(df)
    df = extract_group_dynamics(df)
    df = financial_aggregation(df)
    df = encode_features(df)
    return df

def finalize_for_model(df: pd.DataFrame, drop_cols=None) -> pd.DataFrame:
    if drop_cols is None:
        drop_cols = ["PassengerId", "Name"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])

if __name__ == "__main__":
    from preprocessing import preprocess, TRAIN_PATH, TEST_PATH

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_clean, test_clean, _ = preprocess(train_df, test_df)
    train_feat = engineer_features(train_clean)
    test_feat = engineer_features(test_clean)

    print("Train shape:", train_feat.shape)
    print("Test shape:", test_feat.shape)
    print(train_feat.columns.tolist())