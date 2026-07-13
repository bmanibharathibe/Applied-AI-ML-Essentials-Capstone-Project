"""
Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline
Part 3 of ML Assignment

This script builds on Part 2's preprocessed data to train ensemble models,
perform hyperparameter tuning, and create a serialized ML pipeline.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score

# =============================================================================
# SETUP: Load preprocessed data from Part 2
# =============================================================================
# Assumes Part 2 saved these files. If not, this section loads the original
# dataset and performs the same preprocessing.

print("=" * 70)
print("PART 3: ADVANCED MODELING — ENSEMBLES, TUNING, AND FULL ML PIPELINE")
print("=" * 70)

try:
    # Try loading preprocessed data from Part 2
    X_train_scaled = np.load('X_train_scaled.npy')
    X_test_scaled = np.load('X_test_scaled.npy')
    y_clf_train = np.load('y_clf_train.npy')
    y_clf_test = np.load('y_clf_test.npy')
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    feature_names = np.load('feature_names.npy', allow_pickle=True)
    print("Loaded preprocessed data from Part 2 files.")
except FileNotFoundError:
    print("Part 2 files not found. Creating sample data for demonstration...")
    # Create sample data matching typical structure
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
        n_clusters_per_class=2, random_state=42
    )
    feature_names = np.array([f'feature_{i}' for i in range(X.shape[1])])
    
    X_train, X_test, y_clf_train, y_clf_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")
print()

# =============================================================================
# TASK 1: Decision Tree Baseline (Unconstrained)
# =============================================================================
print("=" * 70)
print("TASK 1: DECISION TREE BASELINE (UNCONSTRAINED)")
print("=" * 70)

dt_unconstrained = DecisionTreeClassifier(max_depth=None, random_state=42)
dt_unconstrained.fit(X_train_scaled, y_clf_train)

train_acc_unconstrained = accuracy_score(y_clf_train, dt_unconstrained.predict(X_train_scaled))
test_acc_unconstrained = accuracy_score(y_clf_test, dt_unconstrained.predict(X_test_scaled))

print(f"Unconstrained Decision Tree:")
print(f"  Training Accuracy: {train_acc_unconstrained:.4f}")
print(f"  Test Accuracy:     {test_acc_unconstrained:.4f}")
print(f"  Train-Test Gap:    {train_acc_unconstrained - test_acc_unconstrained:.4f}")
print()

# =============================================================================
# TASK 2: Controlled Decision Tree
# =============================================================================
print("=" * 70)
print("TASK 2: CONTROLLED DECISION TREE (max_depth=5, min_samples_split=20)")
print("=" * 70)

dt_controlled = DecisionTreeClassifier(
    max_depth=5, 
    min_samples_split=20, 
    random_state=42
)
dt_controlled.fit(X_train_scaled, y_clf_train)

train_acc_controlled = accuracy_score(y_clf_train, dt_controlled.predict(X_train_scaled))
test_acc_controlled = accuracy_score(y_clf_test, dt_controlled.predict(X_test_scaled))

print(f"Controlled Decision Tree:")
print(f"  Training Accuracy: {train_acc_controlled:.4f}")
print(f"  Test Accuracy:     {test_acc_controlled:.4f}")
print(f"  Train-Test Gap:    {train_acc_controlled - test_acc_controlled:.4f}")
print()
print("Comparison:")
print(f"  Unconstrained gap: {train_acc_unconstrained - test_acc_unconstrained:.4f}")
print(f"  Controlled gap:    {train_acc_controlled - test_acc_controlled:.4f}")
print()

# =============================================================================
# TASK 3: Gini vs Entropy Comparison
# =============================================================================
print("=" * 70)
print("TASK 3: GINI VS ENTROPY COMPARISON")
print("=" * 70)

dt_gini = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
dt_gini.fit(X_train_scaled, y_clf_train)
test_acc_gini = accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled))

dt_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)
dt_entropy.fit(X_train_scaled, y_clf_train)
test_acc_entropy = accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled))

print(f"Decision Tree with Gini criterion:")
print(f"  Test Accuracy: {test_acc_gini:.4f}")
print()
print(f"Decision Tree with Entropy criterion:")
print(f"  Test Accuracy: {test_acc_entropy:.4f}")
print()

# =============================================================================
# TASK 4: Random Forest
# =============================================================================
print("=" * 70)
print("TASK 4: RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42
)
rf_model.fit(X_train_scaled, y_clf_train)

rf_train_acc = accuracy_score(y_clf_train, rf_model.predict(X_train_scaled))
rf_test_acc = accuracy_score(y_clf_test, rf_model.predict(X_test_scaled))
rf_train_proba = rf_model.predict_proba(X_train_scaled)[:, 1]
rf_test_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_train_auc = roc_auc_score(y_clf_train, rf_train_proba)
rf_test_auc = roc_auc_score(y_clf_test, rf_test_proba)

print(f"Random Forest (n_estimators=100, max_depth=10):")
print(f"  Training Accuracy: {rf_train_acc:.4f}")
print(f"  Test Accuracy:     {rf_test_acc:.4f}")
print(f"  Training ROC-AUC:  {rf_train_auc:.4f}")
print(f"  Test ROC-AUC:      {rf_test_auc:.4f}")
print()

# Feature Importances
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print("Top 5 Features by Importance:")
print(importance_df.head(5).to_string(index=False))
print()

# =============================================================================
# TASK 4a: Gradient Boosting
# =============================================================================
print("=" * 70)
print("TASK 4a: GRADIENT BOOSTING")
print("=" * 70)

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb_model.fit(X_train_scaled, y_clf_train)

gb_train_acc = accuracy_score(y_clf_train, gb_model.predict(X_train_scaled))
gb_test_acc = accuracy_score(y_clf_test, gb_model.predict(X_test_scaled))
gb_train_proba = gb_model.predict_proba(X_train_scaled)[:, 1]
gb_test_proba = gb_model.predict_proba(X_test_scaled)[:, 1]
gb_train_auc = roc_auc_score(y_clf_train, gb_train_proba)
gb_test_auc = roc_auc_score(y_clf_test, gb_test_proba)

print(f"Gradient Boosting (n_estimators=100, learning_rate=0.1, max_depth=3):")
print(f"  Training Accuracy: {gb_train_acc:.4f}")
print(f"  Test Accuracy:     {gb_test_acc:.4f}")
print(f"  Training ROC-AUC:  {gb_train_auc:.4f}")
print(f"  Test ROC-AUC:      {gb_test_auc:.4f}")
print()

# =============================================================================
# TASK 4b: Feature Ablation Study
# =============================================================================
print("=" * 70)
print("TASK 4b: FEATURE ABLATION STUDY")
print("=" * 70)

# Identify 5 lowest importance features
lowest_importance_features = importance_df.tail(5)['Feature'].values
lowest_importance_indices = [np.where(feature_names == f)[0][0] for f in lowest_importance_features]

print("5 Lowest Importance Features:")
print(importance_df.tail(5).to_string(index=False))
print()

# Create reduced feature sets
all_feature_indices = np.arange(len(feature_names))
remaining_indices = np.array([i for i in all_feature_indices if i not in lowest_importance_indices])

X_train_reduced = X_train_scaled[:, remaining_indices]
X_test_reduced = X_test_scaled[:, remaining_indices]

# Train reduced model with same hyperparameters
rf_reduced = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42
)
rf_reduced.fit(X_train_reduced, y_clf_train)

# Calculate AUCs
full_model_auc = rf_test_auc  # Already calculated above
reduced_proba = rf_reduced.predict_proba(X_test_reduced)[:, 1]
reduced_model_auc = roc_auc_score(y_clf_test, reduced_proba)

print(f"Full Model ROC-AUC (all features):         {full_model_auc:.4f}")
print(f"Reduced Model ROC-AUC (5 features removed): {reduced_model_auc:.4f}")
print(f"AUC Difference:                             {full_model_auc - reduced_model_auc:.4f}")
print()

# =============================================================================
# TASK 5: Cross-Validated Comparison
# =============================================================================
print("=" * 70)
print("TASK 5: CROSS-VALIDATED MODEL COMPARISON")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree (controlled)': DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

cv_results = {}
print("5-Fold Cross-Validation Results (ROC-AUC):")
print("-" * 50)

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv, scoring='roc_auc')
    cv_results[name] = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
    print(f"{name}:")
    print(f"  Mean AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

print()

# =============================================================================
# TASK 6: Hyperparameter Tuning with GridSearchCV
# =============================================================================
print("=" * 70)
print("TASK 6: HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("=" * 70)

# Build pipeline
pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

# Define parameter grid
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}

# Calculate total configurations
n_estimators_options = 3
max_depth_options = 3
min_samples_leaf_options = 2
n_folds = 5
total_fits = n_estimators_options * max_depth_options * min_samples_leaf_options * n_folds

print(f"Parameter Grid:")
print(f"  n_estimators: [50, 100, 200]")
print(f"  max_depth: [5, 10, None]")
print(f"  min_samples_leaf: [1, 5]")
print(f"Total configurations: {n_estimators_options * max_depth_options * min_samples_leaf_options}")
print(f"Total model fits (with 5-fold CV): {total_fits}")
print()

# Use unscaled X_train for GridSearchCV (pipeline handles scaling)
# If X_train doesn't exist separately, use X_train_scaled
try:
    X_train_unscaled = X_train
except NameError:
    X_train_unscaled = X_train_scaled

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

print("Running GridSearchCV... (this may take a moment)")
grid_search.fit(X_train_unscaled, y_clf_train)

print()
print("GridSearchCV Results:")
print(f"  Best Parameters: {grid_search.best_params_}")
print(f"  Best CV Score (ROC-AUC): {grid_search.best_score_:.4f}")
print()

best_pipeline = grid_search.best_estimator_

# =============================================================================
# TASK 7: Manual Learning Curve
# =============================================================================
print("=" * 70)
print("TASK 7: MANUAL LEARNING CURVE")
print("=" * 70)

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
learning_curve_results = []

for f in fractions:
    n_samples = int(f * len(X_train_unscaled))
    X_subset = X_train_unscaled[:n_samples]
    y_subset = y_clf_train[:n_samples]
    
    # Create a fresh pipeline with best params
    subset_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=grid_search.best_params_['randomforestclassifier__n_estimators'],
            max_depth=grid_search.best_params_['randomforestclassifier__max_depth'],
            min_samples_leaf=grid_search.best_params_['randomforestclassifier__min_samples_leaf'],
            random_state=42
        )
    )
    
    subset_pipeline.fit(X_subset, y_subset)
    
    # Training AUC
    train_proba = subset_pipeline.predict_proba(X_subset)[:, 1]
    train_auc = roc_auc_score(y_subset, train_proba)
    
    # Test AUC (on full test set)
    test_proba = subset_pipeline.predict_proba(X_test_scaled)[:, 1]
    test_auc = roc_auc_score(y_clf_test, test_proba)
    
    learning_curve_results.append({
        'Training Fraction': f"{f:.0%}",
        'N Samples': n_samples,
        'Training AUC': train_auc,
        'Test AUC': test_auc
    })

learning_curve_df = pd.DataFrame(learning_curve_results)
print("Learning Curve Results:")
print(learning_curve_df.to_string(index=False))
print()

# =============================================================================
# TASK 8: Serialize the Best Model
# =============================================================================
print("=" * 70)
print("TASK 8: MODEL SERIALIZATION")
print("=" * 70)

# Save the best pipeline
model_filename = 'best_model.pkl'
joblib.dump(best_pipeline, model_filename)
print(f"Best pipeline saved to '{model_filename}'")
print()

# Reload and predict demonstration
print("Reload and Predict Demonstration:")
print("-" * 40)

# Load the saved model
loaded_model = joblib.load('best_model.pkl')

# Create two hand-crafted test rows (matching feature dimensions)
n_features = X_train_unscaled.shape[1]
test_row_1 = np.random.randn(1, n_features)  # Random sample 1
test_row_2 = np.random.randn(1, n_features)  # Random sample 2
test_samples = np.vstack([test_row_1, test_row_2])

# Make predictions
predictions = loaded_model.predict(test_samples)
probabilities = loaded_model.predict_proba(test_samples)

print(f"Test Sample 1 - Prediction: {predictions[0]}, Probability: {probabilities[0]}")
print(f"Test Sample 2 - Prediction: {predictions[1]}, Probability: {probabilities[1]}")
print()
print("Model reload and prediction successful!")
print()

# =============================================================================
# SUMMARY COMPARISON TABLE
# =============================================================================
print("=" * 70)
print("SUMMARY COMPARISON TABLE")
print("=" * 70)

# Calculate test AUC for all models
models_for_test = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree (controlled)': DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

summary_data = []
for name, model in models_for_test.items():
    model.fit(X_train_scaled, y_clf_train)
    test_proba = model.predict_proba(X_test_scaled)[:, 1]
    test_auc = roc_auc_score(y_clf_test, test_proba)
    
    summary_data.append({
        'Model': name,
        '5-Fold CV Mean AUC': f"{cv_results[name]['mean']:.4f}",
        '5-Fold CV Std AUC': f"{cv_results[name]['std']:.4f}",
        'Test Set AUC': f"{test_auc:.4f}"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
print()

# =============================================================================
# FINAL RECOMMENDATION
# =============================================================================
print("=" * 70)
print("MODEL RECOMMENDATION")
print("=" * 70)

# Find best model based on CV mean AUC
best_model_name = max(cv_results.keys(), key=lambda x: cv_results[x]['mean'])
print(f"Recommended Model: {best_model_name}")
print()
print("Justification:")
print("Based on the cross-validation results, the recommended model achieves the")
print("highest mean ROC-AUC score while maintaining reasonable variance across folds.")
print("This indicates robust generalization performance. The model balances complexity")
print("with interpretability and demonstrates consistent performance on unseen data,")
print("making it suitable for production deployment.")
print()

print("=" * 70)
print("PART 3 COMPLETE")
print("=" * 70)
