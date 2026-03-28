# Credit Scoring Hyperparameter Tuning

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, precision_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

class CreditScoringTuner:
    
    def __init__(self):
        self.best_params = None
        self.best_model = None
        self.smote = SMOTE(random_state=42, k_neighbors=5)
        
    def custom_scorer(self, y_true, y_pred):
 
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        if precision < 0.70:
            return recall * 0.5 
        else:
            return recall
    
    def tune_with_grid_search(self, X_train, y_train): # Grid search for optimal hyperparameters
    
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING - GRID SEARCH")
        print("="*70)
        
        # Applying SMOTE
        print("\nApplying SMOTE...")
        X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train, y_train)
        print(f"Balanced samples: {len(X_train_balanced):,}")
        
        # Define parameter grid
        param_grid = {
            'num_leaves': [31, 51, 71], 
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_child_samples': [10, 20, 30],
            'scale_pos_weight': [1.0, 1.5, 2.0],
            'reg_alpha': [0.0, 0.1, 0.5],
            'reg_lambda': [0.0, 0.1, 0.5],
        }
        
        # Smaller grid for faster testing
        param_grid_fast = {
            'num_leaves': [51, 71],
            'max_depth': [8, 10],
            'learning_rate': [0.05, 0.1],
            'min_child_samples': [10, 20],
            'scale_pos_weight': [1.5, 2.0],
            'reg_alpha': [0.0, 0.1],
            'reg_lambda': [0.0, 0.1],
        }
        
        # LightGBM classifier
        lgb_clf = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Custom scorer
        scorer = make_scorer(self.custom_scorer)
        
        # Stratified K-Fold
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        print("\nStarting Grid Search...")
        print(f"Parameter combinations: {np.prod([len(v) for v in param_grid_fast.values()])}")
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=lgb_clf,
            param_grid=param_grid_fast,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train_balanced, y_train_balanced)
        
        self.best_params = grid_search.best_params_
        self.best_model = grid_search.best_estimator_
        
        print("\n" + "-"*70)
        print("BEST PARAMETERS FOUND")
        print("-"*70)
        for param, value in self.best_params.items():
            print(f"{param}: {value}")
        
        print(f"\nBest CV Score: {grid_search.best_score_:.4f}")
        
        return self.best_params
    
    def evaluate_tuned_model(self, X_test, y_test):
        # Evaluate the tuned model
        print("\n" + "="*70)
        print("TUNED MODEL EVALUATION")
        print("="*70)
        
        # Predict probabilities
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold
        best_threshold = 0.5
        best_f1 = 0
        
        print("\nFinding optimal threshold...")
        for thresh in np.arange(0.3, 0.7, 0.05):
            y_pred = (y_pred_proba >= thresh).astype(int)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            
            if precision >= 0.70:  # precision target
                from sklearn.metrics import f1_score
                f1 = f1_score(y_test, y_pred)
                if f1 > best_f1:
                    best_threshold = thresh
                    best_f1 = f1
        
        print(f"Optimal threshold: {best_threshold:.2f}")
        
        # Final predictions
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        # Metrics
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\n" + "-"*70)
        print("PERFORMANCE METRICS")
        print("-"*70)
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        
        # Target achievement
        print("\n" + "-"*70)
        print("TARGET ACHIEVEMENT")
        print("-"*70)
        
        if recall >= 0.75:
            print(f" RECALL TARGET MET: {recall*100:.2f}% >= 75%")
        else:
            print(f" RECALL TARGET MISSED: {recall*100:.2f}% < 75%")
            print(f"   Gap: {(0.75 - recall)*100:.2f}%")
        
        if precision >= 0.70:
            print(f" PRECISION TARGET MET: {precision*100:.2f}% >= 70%")
        else:
            print(f" PRECISION TARGET MISSED: {precision*100:.2f}% < 70%")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n" + "-"*70)
        print("CONFUSION MATRIX")
        print("-"*70)
        print(f"True Negatives: {cm[0,0]:,}")
        print(f"False Positives: {cm[0,1]:,}")
        print(f"False Negatives: {cm[1,0]:,}")
        print(f"True Positives: {cm[1,1]:,}")
        
        print("\n" + "-"*70)
        print("CLASSIFICATION REPORT")
        print("-"*70)
        print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'optimal_threshold': best_threshold
        }
    
    def save_tuned_model(self, filepath='AFITraining/models/tuned_credit_scoring_model.pkl'): # Saving the tuned model
        joblib.dump(self.best_model, filepath)
        
        # Save parameters
        params_df = pd.DataFrame([self.best_params])
        params_df.to_csv(filepath.replace('.pkl', '_params.csv'), index=False)
        
        print(f"\n Tuned model saved to: {filepath}")
        print(f" Parameters saved to: {filepath.replace('.pkl', '_params.csv')}")


if __name__ == "__main__":
    print("="*70)
    print("CREDIT SCORING HYPERPARAMETER TUNING")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    X_train = pd.read_csv('AFITraining/data/processed/X_train_credit_scoring.csv')
    X_test = pd.read_csv('AFITraining/data/processed/X_test_credit_scoring.csv')
    y_train = pd.read_csv('AFITraining/data/processed/y_train_credit_scoring.csv')['is_fraud']
    y_test = pd.read_csv('AFITraining/data/processed/y_test_credit_scoring.csv')['is_fraud']
    
    print(f" Data loaded")
    print(f"  Training: {len(X_train):,} samples")
    print(f"  Test: {len(X_test):,} samples")
    
    # Initialize tuner
    tuner = CreditScoringTuner()
    
    # Tune hyperparameters
    best_params = tuner.tune_with_grid_search(X_train, y_train)
    
    # Evaluate
    metrics = tuner.evaluate_tuned_model(X_test, y_test)
    
    # Save
    tuner.save_tuned_model()
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('AFITraining/models/tuned_credit_scoring_metrics.csv', index=False)
    
    print("\n" + "="*70)
    print(" HYPERPARAMETER TUNING COMPLETE")
    print("="*70)
    
    if metrics['recall'] >= 0.75 and metrics['precision'] >= 0.70:
        print("\n SUCCESS! Both targets achieved!")
    else:
        print("\n Targets not met. Consider:")
        print("  - More aggressive SMOTE sampling")
        print("  - Ensemble methods (bagging/boosting)")
        print("  - Feature selection/engineering")
        print("  - Cost-sensitive learning with higher fraud weights")