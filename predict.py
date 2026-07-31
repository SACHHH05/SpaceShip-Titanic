import os
import pandas as pd

from preprocessing import preprocess, TRAIN_PATH, TEST_PATH
from features import engineer_features, finalize_for_model
from train import build_model, OUTPUT_DIR


def run():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_clean, test_clean, _ = preprocess(train_df, test_df)
    train_feat = engineer_features(train_clean)
    test_feat = engineer_features(test_clean)

    y = train_feat["Transported"].astype(int)
    X = finalize_for_model(train_feat, drop_cols=["PassengerId", "Name", "Transported"])

    test_ids = test_feat["PassengerId"]
    X_test = finalize_for_model(test_feat, drop_cols=["PassengerId", "Name"])
    X_test = X_test.reindex(columns=X.columns, fill_value=0)

    model = build_model()
    model.fit(X, y)
    preds = model.predict(X_test).astype(bool)

    submission = pd.DataFrame({"PassengerId": test_ids, "Transported": preds})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/submission.csv"
    submission.to_csv(out_path, index=False)

    print(f"Wrote {len(submission)} predictions to {out_path}")
    print(submission.head())
    return submission


if __name__ == "__main__":
    run()