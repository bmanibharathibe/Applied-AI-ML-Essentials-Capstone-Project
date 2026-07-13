"""
part2_task1_task2.py
---------------------
Part 2 — Supervised Machine Learning Model
Task 1: Load cleaned_data.csv and define X, y_reg, y_clf
Task 2: Encode categorical columns (label encoding for ordinal, one-hot for nominal)

Input : cleaned_data.csv (produced by Part 1 script)
Output: X_encoded, y_reg, y_clf  (ready for Task 3 — train/test split + scaling)
"""

import numpy as np
import pandas as pd

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

# ---------------------------------------------------------------------------
# TASK 1 — Load cleaned_data.csv and define X, y_reg, y_clf
# ---------------------------------------------------------------------------
section("TASK 1 — Load data & define feature matrix / labels")

df = pd.read_csv("cleaned_data.csv")
print("Loaded cleaned_data.csv with shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nDtypes:\n", df.dtypes)

# Safety net: satisfaction_score had >20% nulls in Part 1 (left unfilled there).
# It cannot be used as a model feature with nulls remaining, so we median-impute
# it here if any nulls are still present.
if df["satisfaction_score"].isnull().sum() > 0:
    med = df["satisfaction_score"].median()
    n_missing = df["satisfaction_score"].isnull().sum()
    df["satisfaction_score"] = df["satisfaction_score"].fillna(med)
    print(f"\nFilled {n_missing} remaining nulls in 'satisfaction_score' with median = {med:.2f}")

# join_date isn't directly usable as a numeric/categorical model feature, so we
# engineer a numeric 'tenure_days' feature from it (days since joining, relative
# to the most recent join_date in the dataset) and drop the raw date column.
df["join_date"] = pd.to_datetime(df["join_date"])
reference_date = df["join_date"].max()
df["tenure_days"] = (reference_date - df["join_date"]).dt.days
df = df.drop(columns=["join_date"])
print(f"\nEngineered 'tenure_days' from join_date (reference date = {reference_date.date()}), "
      "dropped raw join_date column.")

# --- Regression label ---
# y_reg: monthly_salary (continuous numeric column)
y_reg = df["monthly_salary"].copy()

# --- Classification label ---
# y_clf: 'attrition' is a natural binary column already present in the dataset
# (Yes/No -> employee left the company or not), so we use it directly rather
# than artificially binarizing y_reg at its median.
y_clf = df["attrition"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
assert y_clf.isnull().sum() == 0, "Unexpected category found in 'attrition' column."
y_clf = y_clf.astype(int)

print("\ny_reg (monthly_salary) summary:\n", y_reg.describe())
print("\ny_clf (attrition) value counts:\n", y_clf.value_counts())
print("y_clf class balance (%):\n", (y_clf.value_counts(normalize=True) * 100).round(2))

# --- Feature matrix X ---
# Drop identifier column, both raw target columns, and any other non-feature
# columns so neither target leaks into X.
drop_cols = ["employee_id", "monthly_salary", "attrition"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

print("\nFeature matrix X shape:", X.shape)
print("X columns:\n", X.columns.tolist())
print("\nX dtypes:\n", X.dtypes)

# ---------------------------------------------------------------------------
# TASK 2 — Encode categorical columns
# ---------------------------------------------------------------------------
section("TASK 2 — Encode categorical columns")

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical columns detected in X:", categorical_cols)

# 'department' -> NOMINAL (no natural order: e.g. Sales, Engineering, HR, Finance
# are just labels, not ranked categories). One-hot encoding is used instead of
# label encoding because label encoding would assign arbitrary integers
# (e.g. Sales=0, Engineering=1, HR=2) that imply a false ordinal relationship
# ("HR > Sales") which the model could wrongly learn a magnitude/ordering
# effect from. One-hot encoding removes that false ordering by giving each
# category its own independent binary column.
if "department" in categorical_cols:
    print("\nUnique values in 'department':", df["department"].unique().tolist())
    X = pd.get_dummies(X, columns=["department"], prefix="dept", drop_first=True)
    print("One-hot encoded 'department' (drop_first=True to avoid multicollinearity).")

# Re-check for any remaining categorical columns that might have a natural
# order (ordinal). None are present in this dataset after department is
# encoded and attrition/monthly_salary were removed as targets, but the
# pattern below is kept as a template in case an ordinal column (e.g. a
# performance rating Low/Medium/High) exists in your dataset.
remaining_categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
if remaining_categorical:
    print("\nRemaining categorical columns (checking for ordinal structure):", remaining_categorical)
    # Example pattern for an ordinal column, uncomment/adjust if applicable:
    # ordinal_map = {"Low": 0, "Medium": 1, "High": 2}
    # X["performance_rating"] = X["performance_rating"].map(ordinal_map)
else:
    print("\nNo remaining categorical columns — all columns in X are now numeric.")

print("\nFinal encoded X shape:", X.shape)
print("Final X columns:\n", X.columns.tolist())
print("\nFinal X dtypes:\n", X.dtypes)
print("\nFirst 5 rows of encoded X:\n", X.head())

# X, y_reg, y_clf are now ready to be passed into Task 3
# (train_test_split + StandardScaler fit only on training data).
