import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss,
)
from xgboost import XGBClassifier

from preprocessing import preprocess, TRAIN_PATH, TEST_PATH, RANDOM_SEED
from features import engineer_features, finalize_for_model

OUTPUT_DIR = "outputs"

def build_model():
    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

def load_training_features():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_clean, test_clean, _ = preprocess(train_df, test_df)
    train_feat = engineer_features(train_clean)

    y = train_feat["Transported"].astype(int)
    X = finalize_for_model(train_feat, drop_cols=["PassengerId", "Name", "Transported"])
    return X, y

def run():
    feature_summary = (
        "Cabin split into Cabin_Deck/Cabin_Number/Cabin_Side; PassengerId group "
        "prefixes produced Group_Size and Is_Alone; the five amenity spend columns "
        "were summed into Total_Spend with a derived Zero_Spend flag; remaining "
        "categoricals were one-hot encoded."
    )

    X, y = load_training_features()
    final_feature_count = X.shape[1]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = []

    start_time = time.time()
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        fold_model = build_model()
        fold_model.fit(X_tr, y_tr)
        fold_pred = fold_model.predict(X_val)
        fold_acc = accuracy_score(y_val, fold_pred)
        cv_scores.append(fold_acc)
        print(f"Fold {fold_idx}: accuracy = {fold_acc:.4f}")

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    final_model = build_model()
    final_model.fit(X_train, y_train)
    training_time_sec = time.time() - start_time

    holdout_pred = final_model.predict(X_holdout)
    holdout_proba = final_model.predict_proba(X_holdout)[:, 1]

    metrics = {
        "random_seed": RANDOM_SEED,
        "final_feature_count": int(final_feature_count),
        "feature_engineering_summary": feature_summary,
        "cv_n_splits": 5,
        "cv_fold_accuracies": [round(s, 6) for s in cv_scores],
        "cv_accuracy_mean": round(cv_mean, 6),
        "cv_accuracy_std": round(cv_std, 6),
        "holdout_accuracy": round(float(accuracy_score(y_holdout, holdout_pred)), 6),
        "holdout_precision": round(float(precision_score(y_holdout, holdout_pred)), 6),
        "holdout_recall": round(float(recall_score(y_holdout, holdout_pred)), 6),
        "holdout_f1": round(float(f1_score(y_holdout, holdout_pred)), 6),
        "holdout_roc_auc": round(float(roc_auc_score(y_holdout, holdout_proba)), 6),
        "holdout_log_loss": round(float(log_loss(y_holdout, holdout_proba)), 6),
        "training_time_seconds": round(training_time_sec, 3),
        "model": "XGBClassifier (n_estimators=400, max_depth=5, lr=0.03)",
        "train_rows": int(X_train.shape[0]),
        "holdout_rows": int(X_holdout.shape[0]),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{OUTPUT_DIR}/feature_columns.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)
    final_model.save_model(f"{OUTPUT_DIR}/model.json")

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics

if __name__ == "__main__":
    run()