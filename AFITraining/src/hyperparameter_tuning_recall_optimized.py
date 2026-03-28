"""
AFI - Hyperparameter Tuning for Credit Scoring
Goal: Maximize RECALL (target: 75%+) while maintaining precision >= 70%
Strategy: Use Optuna for Bayesian optimization with recall-focused objective
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("AFI - AGGRESSIVE HYPERPARAMETER TUNING FOR RECALL OPTIMIZATION")
print("="*80)


# 1. LOAD DATA
# ============================================================================

print("\n[1/5] Loading data...")
X_train = pd.read_csv('AFITraining/data/processed/X_train_credit_scoring.csv')
y_train = pd.read_csv('AFITraining/data/processed/y_train_credit_scoring.csv')['is_fraud']
X_test = pd.read_csv('AFITraining/data/processed/X_test_credit_scoring.csv')
y_test = pd.read_csv('AFITraining/data/processed/y_test_credit_scoring.csv')['is_fraud']

print(f" Training set: {len(X_train):,} samples")
print(f" Test set: {len(X_test):,} samples")
print(f" Fraud rate (train): {y_train.mean()*100:.2f}%")


# 2. APPLY SMOTE
# ============================================================================

print("\n[2/5] Applying SMOTE for class balance...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f" Before SMOTE: {len(X_train):,} samples")
print(f" After SMOTE: {len(X_train_smote):,} samples")
print(f" New fraud rate: {y_train_smote.mean()*100:.2f}%")


# 3. SCALE FEATURES
# ============================================================================

print("\n[3/5] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(" Features scaled")


# 4. OPTUNA OPTIMIZATION
# ============================================================================

print("\n[4/5] Starting Optuna hyperparameter optimization...")
print("Objective: Maximize RECALL while keeping Precision >= 70%")

def objective(trial):
    
    # Optuna objective function
    # Suggest hyperparameters with AGGRESSIVE settings for recall
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        
        # CRITICAL PARAMETERS FOR RECALL
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 3.0, 10.0),
        
        # Tree structure
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1e-2, log=True),
        
        # Learning
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        
        # Regularization 
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        
        # Sampling
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'subsample_freq': 1,
    }
    
    # 5 fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    recalls = []
    precisions = []
    
    for train_idx, val_idx in skf.split(X_train_scaled, y_train_smote):
        X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
        y_tr, y_val = y_train_smote.iloc[train_idx], y_train_smote.iloc[val_idx]
        
        # Train model
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
        
        # Predict
        y_pred = model.predict(X_val)
        
        # Metrics
        recall = recall_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        
        recalls.append(recall)
        precisions.append(precision)
    
    avg_recall = np.mean(recalls)
    avg_precision = np.mean(precisions)
    
    # PENALTY if precision drops below 70%
    if avg_precision < 0.70:
        penalty = (0.70 - avg_precision) * 2  # Heavy penalty
        return avg_recall - penalty
    
    # Otherwise, maximize recall
    return avg_recall


print("\nStarting optimization (50 trials)...")

study = optuna.create_study(direction='maximize', study_name='recall_optimization')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)

best_params = study.best_params
print(f"\nBest Recall Score (CV): {study.best_value*100:.2f}%")
print(f"\nBest Hyperparameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")


# 5. TRAIN FINAL MODEL WITH BEST PARAMETERS
# ============================================================================

print("\n[5/5] Training final model with best parameters...")

# Add fixed parameters
final_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
    **best_params
}

# Train on full training set
final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(
    X_train_scaled, 
    y_train_smote,
    eval_set=[(X_test_scaled, y_test)],
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

# 6. EVALUATE ON TEST SET
# ============================================================================

print("\n" + "="*80)
print("FINAL MODEL EVALUATION ON TEST SET")
print("="*80)

y_pred = final_model.predict(X_test_scaled)
y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
accuracy = (y_pred == y_test).mean()

print(f"\nPerformance Metrics:")
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1-Score:  {f1:.4f}")
print(f"  AUC-ROC:   {auc:.4f}")

print(f"\n" + "="*80)
print("TARGET ACHIEVEMENT")
print("="*80)

# Check targets
if recall >= 0.75:
    print(f" RECALL TARGET MET: {recall*100:.2f}% >= 75%")
else:
    print(f" RECALL TARGET MISSED: {recall*100:.2f}% < 75% (Gap: {(0.75-recall)*100:.2f}%)")

if precision >= 0.70:
    print(f" PRECISION TARGET MET: {precision*100:.2f}% >= 70%")
else:
    print(f"  PRECISION BELOW TARGET: {precision*100:.2f}% < 70% (Gap: {(0.70-precision)*100:.2f}%)")


# 7. SAVE MODEL AND RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODEL AND RESULTS")
print("="*80)

# Save model
joblib.dump(final_model, 'AFITraining/models/optimized_credit_scoring_model.pkl')
print(" Model saved: AFITraining/models/optimized_credit_scoring_model.pkl")

# Save scaler
joblib.dump(scaler, 'AFITraining/models/optimized_scaler.pkl')
print(" Scaler saved: AFITraining/models/optimized_scaler.pkl")

# Save metrics
metrics = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'auc_roc': float(auc),
    'best_params': best_params,
    'cv_recall': float(study.best_value)
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('AFITraining/models/optimized_credit_scoring_metrics.csv', index=False)
print(" Metrics saved: AFITraining/models/optimized_credit_scoring_metrics.csv")

# Save best parameters
import json
with open('AFITraining/models/best_hyperparameters.json', 'w') as f:
    json.dump(best_params, f, indent=2)
print(" Best parameters saved: AFITraining/models/best_hyperparameters.json")

# Save optuna study
joblib.dump(study, 'AFITraining/models/optuna_study.pkl')
print(" Optuna study saved: AFITraining/models/optuna_study.pkl")

print("\n" + "="*80)
print(" OPTIMIZATION COMPLETE!")
print("="*80)

print("\nFiles created:")
print("  - AFITraining/models/optimized_credit_scoring_model.pkl")
print("  - AFITraining/models/optimized_scaler.pkl")
print("  - AFITraining/models/optimized_credit_scoring_metrics.csv")
print("  - AFITraining/models/best_hyperparameters.json")
print("  - AFITraining/models/optuna_study.pkl")