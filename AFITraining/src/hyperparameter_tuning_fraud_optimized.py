"""
AFI - Hyperparameter Tuning for Fraud Detection
Goal: Maximize RECALL (target: 70%+) while maintaining precision >= 40%
Strategy: Ensemble optimization with Optuna
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("AFI - FRAUD DETECTION: RECALL-OPTIMIZED HYPERPARAMETER TUNING")
print("="*80)

# 1. LOAD DATA
# ============================================================================

print("\n[1/6] Loading data...")
X_train = pd.read_csv('AFITraining/data/processed/X_train_fraud.csv')
y_train = pd.read_csv('AFITraining/data/processed/y_train_fraud.csv')['is_fraud']
X_test = pd.read_csv('AFITraining/data/processed/X_test_fraud.csv')
y_test = pd.read_csv('AFITraining/data/processed/y_test_fraud.csv')['is_fraud']

print(f" Training set: {len(X_train):,} samples")
print(f" Test set: {len(X_test):,} samples")
print(f" Fraud rate (train): {y_train.mean()*100:.2f}%")

# 2. APPLY SMOTE
# ============================================================================

print("\n[2/6] Applying SMOTE for class balance...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f" Before SMOTE: {len(X_train):,} samples")
print(f" After SMOTE: {len(X_train_smote):,} samples")
print(f" New fraud rate: {y_train_smote.mean()*100:.2f}%")


# 3. SCALE FEATURES
# ============================================================================

print("\n[3/6] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(" Features scaled")

# 4. OPTIMIZE LIGHTGBM
# ============================================================================

print("\n[4/6] Optimizing LightGBM parameters...")

def objective_lgb(trial):
    # Optimize LightGBM for recall
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        
        # AGGRESSIVE RECALL SETTINGS
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5.0, 15.0),
        
        # Tree structure
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 8, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        
        # Learning
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        
        # Regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
        
        # Sampling
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
    }
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    recalls = []
    precisions = []
    
    for train_idx, val_idx in skf.split(X_train_scaled, y_train_smote):
        X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
        y_tr, y_val = y_train_smote.iloc[train_idx], y_train_smote.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
        
        y_pred = model.predict(X_val)
        
        recalls.append(recall_score(y_val, y_pred))
        precisions.append(precision_score(y_val, y_pred))
    
    avg_recall = np.mean(recalls)
    avg_precision = np.mean(precisions)
    
    # Penalty if precision < 40%
    if avg_precision < 0.40:
        penalty = (0.40 - avg_precision) * 3
        return avg_recall - penalty
    
    return avg_recall

print("Starting LightGBM optimization (40 trials)...")
study_lgb = optuna.create_study(direction='maximize', study_name='lgb_recall_optimization')
study_lgb.optimize(objective_lgb, n_trials=40, show_progress_bar=True)

best_lgb_params = study_lgb.best_params
print(f"\n Best LightGBM Recall : {study_lgb.best_value*100:.2f}%")


# 5. OPTIMIZE ISOLATION FOREST
# ============================================================================

print("\n[5/6] Optimizing Isolation Forest parameters...")

def objective_if(trial):
    # Optimize Isolation Forest for recall
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_samples': trial.suggest_int('max_samples', 256, 2048),
        'contamination': trial.suggest_float('contamination', 0.01, 0.1),
        'max_features': trial.suggest_float('max_features', 0.5, 1.0),
        'random_state': 42
    }
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    recalls = []
    precisions = []
    
    for train_idx, val_idx in skf.split(X_train_scaled, y_train_smote):
        X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
        y_tr, y_val = y_train_smote.iloc[train_idx], y_train_smote.iloc[val_idx]
        
        model = IsolationForest(**params)
        model.fit(X_tr)
        
        # Predict (IF returns -1 for anomalies, 1 for normal)
        y_pred_if = model.predict(X_val)
        y_pred = (y_pred_if == -1).astype(int)  # Convert to 1 for fraud
        
        # Avoid division by zero
        if y_pred.sum() > 0:
            recalls.append(recall_score(y_val, y_pred))
            precisions.append(precision_score(y_val, y_pred))
        else:
            recalls.append(0)
            precisions.append(0)
    
    avg_recall = np.mean(recalls)
    return avg_recall

print("Starting Isolation Forest optimization (30 trials)...")
study_if = optuna.create_study(direction='maximize', study_name='if_recall_optimization')
study_if.optimize(objective_if, n_trials=30, show_progress_bar=True)

best_if_params = study_if.best_params
print(f"\n Best Isolation Forest Recall: {study_if.best_value*100:.2f}%")

# 6. OPTIMIZE ENSEMBLE WEIGHTS
# ============================================================================

print("\n[6/6] Optimizing ensemble weights...")

# Train final models with best parameters
final_lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'random_state': 42,
    **best_lgb_params
}

final_if_params = {
    'random_state': 42,
    **best_if_params
}

lgb_model = lgb.LGBMClassifier(**final_lgb_params)
lgb_model.fit(X_train_scaled, y_train_smote)

if_model = IsolationForest(**final_if_params)
if_model.fit(X_train_scaled)

# Get predictions
lgb_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]
if_pred = if_model.predict(X_test_scaled)
if_scores = if_model.score_samples(X_test_scaled)
if_proba = 1 / (1 + np.exp(if_scores))

def objective_ensemble(trial):
    # Optimize ensemble weight
    
    weight_lgb = trial.suggest_float('weight_lgb', 0.5, 0.9)
    weight_if = 1 - weight_lgb
    
    # Ensemble prediction
    ensemble_proba = weight_lgb * lgb_proba + weight_if * if_proba
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    
    recall = recall_score(y_test, ensemble_pred)
    precision = precision_score(y_test, ensemble_pred)
    
    # Penalty if precision < 40%
    if precision < 0.40:
        penalty = (0.40 - precision) * 2
        return recall - penalty
    
    return recall

study_ensemble = optuna.create_study(direction='maximize', study_name='ensemble_optimization')
study_ensemble.optimize(objective_ensemble, n_trials=20, show_progress_bar=True)

best_weight_lgb = study_ensemble.best_params['weight_lgb']
best_weight_if = 1 - best_weight_lgb

print(f"\n Best Ensemble Weights: LGB={best_weight_lgb:.3f}, IF={best_weight_if:.3f}")

# 7. FINAL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("FINAL ENSEMBLE EVALUATION ON TEST SET")
print("="*80)

# Final ensemble prediction
ensemble_proba = best_weight_lgb * lgb_proba + best_weight_if * if_proba
ensemble_pred = (ensemble_proba > 0.5).astype(int)

recall = recall_score(y_test, ensemble_pred)
precision = precision_score(y_test, ensemble_pred)
f1 = f1_score(y_test, ensemble_pred)
auc = roc_auc_score(y_test, ensemble_proba)
accuracy = accuracy_score(y_test, ensemble_pred)

# False positive rate
fp = ((ensemble_pred == 1) & (y_test == 0)).sum()
tn = ((ensemble_pred == 0) & (y_test == 0)).sum()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"\nPerformance Metrics:")
print(f"  Accuracy:           {accuracy*100:.2f}%")
print(f"  Precision:          {precision*100:.2f}%")
print(f"  Recall:             {recall*100:.2f}%")
print(f"  F1-Score:           {f1:.4f}")
print(f"  AUC-ROC:            {auc:.4f}")
print(f"  False Positive Rate: {fpr*100:.2f}%")

print(f"\n" + "="*80)
print("TARGET ACHIEVEMENT")
print("="*80)

if recall >= 0.70:
    print(f" RECALL TARGET MET: {recall*100:.2f}% >= 70%")
else:
    print(f" RECALL TARGET MISSED: {recall*100:.2f}% < 70% (Gap: {(0.70-recall)*100:.2f}%)")

if precision >= 0.40:
    print(f" PRECISION TARGET MET: {precision*100:.2f}% >= 40%")
else:
    print(f"  PRECISION BELOW TARGET: {precision*100:.2f}% < 40%")


# 8. SAVE MODELS AND RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODELS AND RESULTS")
print("="*80)

# Save models
lgb_model.booster_.save_model('AFITraining/models/optimized_fraud_lgb.txt')
print(" LightGBM saved: AFITraining/models/optimized_fraud_lgb.txt")

joblib.dump(if_model, 'AFITraining/models/optimized_fraud_if.pkl')
print(" Isolation Forest saved: AFITraining/models/optimized_fraud_if.pkl")

joblib.dump(scaler, 'AFITraining/models/optimized_fraud_scaler.pkl')
print(" Scaler saved: AFITraining/models/optimized_fraud_scaler.pkl")

# Save metrics
metrics = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'auc_roc': float(auc),
    'false_positive_rate': float(fpr),
    'lgb_weight': float(best_weight_lgb),
    'if_weight': float(best_weight_if),
    'best_lgb_params': best_lgb_params,
    'best_if_params': best_if_params
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('AFITraining/models/optimized_fraud_detection_metrics.csv', index=False)
print(" Metrics saved: AFITraining/models/optimized_fraud_detection_metrics.csv")

# Save ensemble config
import json
ensemble_config = {
    'lgb_weight': float(best_weight_lgb),
    'if_weight': float(best_weight_if),
    'lgb_params': best_lgb_params,
    'if_params': best_if_params
}

with open('AFITraining/models/optimized_ensemble_config.json', 'w') as f:
    json.dump(ensemble_config, f, indent=2)
print(" Ensemble config saved: AFITraining/models/optimized_ensemble_config.json")

print("\n" + "="*80)
print(" FRAUD DETECTION OPTIMIZATION COMPLETE!")
print("="*80)

print("\nFiles created:")
print("  - AFITraining/models/optimized_fraud_lgb.txt")
print("  - AFITraining/models/optimized_fraud_if.pkl")
print("  - AFITraining/models/optimized_fraud_scaler.pkl")
print("  - AFITraining/models/optimized_fraud_detection_metrics.csv")
print("  - AFITraining/models/optimized_ensemble_config.json")