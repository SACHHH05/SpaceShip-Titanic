# SpaceShip-Titanic

Binary classification pipeline predicting the `Transported` status of
passengers for the Kaggle "Spaceship Titanic" competition.

## Project Structure
```
SpaceShip-Titanic/
├── dataset/
│   ├── train.csv
│   └── test.csv
├── preprocessing.py       # missing-value imputation
├── features.py             # feature engineering
├── train.py                 # CV, training, metric extraction
├── predict.py                # test-set inference -> submission.csv
├── build_report.py            # builds the Excel metrics workbook
├── requirements.txt
└── outputs/
    ├── metrics.json
    ├── feature_columns.json
    ├── model.json
    ├── submission.csv
    └── ML_Assignment_Submission_and_Validation.xlsx
```

## Pipeline

1. **Preprocessing** (`preprocessing.py`)
   - Numeric columns (`Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`) → filled with the **median**, computed from `train.csv` only.
   - Categorical columns (`HomePlanet`, `CryoSleep`, `Destination`, `VIP`) → filled with the **mode**, computed from `train.csv` only.
   - `Cabin` → forward/backward-filled within each passenger's travel group (same `PassengerId` prefix), falling back to `"Unknown/0/Unknown"`.
   - Same fill values learned from train are applied to test — no data leakage.

2. **Feature Engineering** (`features.py`)
   - `Cabin` → `Cabin_Deck`, `Cabin_Number`, `Cabin_Side`
   - `PassengerId` group → `Group_Size`, `Is_Alone`
   - Amenity spend → `Total_Spend`, `Zero_Spend` (strong proxy for `CryoSleep`)
   - Categorical text columns one-hot encoded; booleans cast to 0/1

3. **Validation & Training** (`train.py`)
   - 80/20 train/held-out split
   - 5-fold `StratifiedKFold` (`shuffle=True`, `random_state=42`) on the 80% split
   - Model: `XGBClassifier` (400 trees, max_depth=5, learning_rate=0.03)
   - Writes `outputs/metrics.json`, `outputs/model.json`, `outputs/feature_columns.json`

4. **Prediction** (`predict.py`)
   - Retrains on the full training set, predicts on `test.csv`
   - Writes `outputs/submission.csv` (`PassengerId`, `Transported`)

5. **Report** (`build_report.py`)
   - Builds `outputs/ML_Assignment_Submission_and_Validation.xlsx` from `metrics.json`
   - CV mean/std computed as live Excel formulas from the 5 fold accuracies

## Reproducing

```bash
pip install -r requirements.txt
python preprocessing.py   # sanity check: missing values before/after
python train.py           # trains model, writes outputs/metrics.json, outputs/model.json
python predict.py         # writes outputs/submission.csv
python build_report.py    # writes outputs/ML_Assignment_Submission_and_Validation.xlsx
```

## Results (random_seed = 42)

| Metric | Value |
|---|---|
| Model | XGBoost (XGBClassifier) |
| Final feature count | 31 |
| Missing-value handling | Median (numeric) / mode (categorical) from train only; group-based fill for Cabin |
| CV strategy | StratifiedKFold, n_splits=5, shuffle=True, random_state=42 |
| CV Accuracy (mean) | 0.8070 |
| CV Accuracy (std) | 0.0060 |
| Validation Accuracy (20% held-out) | 0.8125 |
| Precision | 0.8118 |
| Recall | 0.8174 |
| F1 Score | 0.8146 |
| ROC-AUC | 0.9131 |
| Log Loss | 0.3640 |
| Random seed set? | Y (42) |
| Training time | ~2.8s |

Full detail: `outputs/metrics.json` and `outputs/ML_Assignment_Submission_and_Validation.xlsx`.
