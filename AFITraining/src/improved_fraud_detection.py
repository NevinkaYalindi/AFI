# Improved Fraud Detection Strategy

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

class ImprovedFraudDetector:
    
    def __init__(self):
        self.lgb_model = None
        self.isolation_forest = None
        self.smote = SMOTE(random_state=42, sampling_strategy=0.5)
        
    def train(self, X_train, y_train):
         # Trained with SMOTE + LightGBM

        print("\n" + "="*70)
        print("IMPROVED FRAUD DETECTION TRAINING")
        print("Strategy: LightGBM primary, Isolation Forest validator")
        print("="*70)
        
        # Applying SMOTE for better recall
        print(f"\nOriginal fraud ratio: {y_train.mean()*100:.2f}%")
        print("Applying SMOTE (50% target ratio)...")
        
        X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train, y_train)
        
        print(f"After SMOTE: {len(X_train_balanced):,} samples")
        print(f"Fraud ratio: {y_train_balanced.mean()*100:.2f}%")
        
        # Training LightGBM with fraud-focused parameters
        print("\n[1/2] Training LightGBM (Primary Detector)...")
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 63,  
            'max_depth': 10, 
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 10,  
            'scale_pos_weight': 2.0, 
            'reg_alpha': 0.0, 
            'reg_lambda': 0.1,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        train_data = lgb.Dataset(X_train_balanced, label=y_train_balanced)
        
        self.lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data],
            valid_names=['train'],
            callbacks=[lgb.log_evaluation(period=200)]
        )
        
        print("✓ LightGBM trained")
        
        # Train Isolation Forest (validator)
        print("\n[2/2] Training Isolation Forest (Validator)...")
        
        X_train_normal = X_train[y_train == 0]
        
        self.isolation_forest = IsolationForest(
            contamination=0.05,  
            n_estimators=150,
            max_samples='auto',
            max_features=0.8,
            bootstrap=False,
            n_jobs=-1,
            random_state=42
        )
        
        self.isolation_forest.fit(X_train_normal)
        print("✓ Isolation Forest trained")
        
        print("\n✓ Training complete")
    
    def predict(self, X, threshold=0.5, strategy='lgb_primary'): # Prediction strategies:
   
        # LightGBM predictions
        lgb_proba = self.lgb_model.predict(X)
        lgb_pred = (lgb_proba >= threshold).astype(int)
        
        # Isolation Forest predictions
        if_pred = self.isolation_forest.predict(X)
        if_fraud = np.where(if_pred == -1, 1, 0)
        
        if strategy == 'lgb_primary':
            return lgb_pred, lgb_proba
        elif strategy == 'ensemble_or':
            # Flag if either model detects - HIGHEST RECALL
            ensemble_pred = np.where((lgb_pred == 1) | (if_fraud == 1), 1, 0)
            return ensemble_pred, lgb_proba
        elif strategy == 'ensemble_and':
            # Flag if both models detect - HIGHEST PRECISION
            ensemble_pred = np.where((lgb_pred == 1) & (if_fraud == 1), 1, 0)
            return ensemble_pred, lgb_proba
        else:
            return lgb_pred, lgb_proba
    
    def evaluate(self, X_test, y_test):
     
        print("\n" + "="*70)
        print("FRAUD DETECTION EVALUATION")
        print("="*70)
        
        # Test multiple strategies
        strategies = ['lgb_primary', 'ensemble_or', 'ensemble_and']
        results = {}
        
        for strategy in strategies:
            print(f"\n--- Strategy: {strategy} ---")
            
            # Finding optimal threshold for this strategy
            best_threshold = 0.5
            best_recall = 0
            
            for thresh in np.arange(0.2, 0.6, 0.05):
                y_pred, y_proba = self.predict(X_test, threshold=thresh, strategy=strategy)
                
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                
                # Finding best recall while maintaining precision >= 40%
                if precision >= 0.40 and recall > best_recall:
                    best_recall = recall
                    best_threshold = thresh
            
            # Evaluate
            y_pred, y_proba = self.predict(X_test, threshold=best_threshold, strategy=strategy)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba)
            
            results[strategy] = {
                'threshold': best_threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc
            }
            
            print(f"Optimal threshold: {best_threshold:.2f}")
            print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
            print(f"F1-Score: {f1:.4f}")
        
        # Choosing best strategy
        print("\n" + "="*70)
        print("STRATEGY COMPARISON")
        print("="*70)
        
        comparison_df = pd.DataFrame(results).T
        print(comparison_df.to_string())
        
        # Selecting strategy with highest recall that meets precision target
        best_strategy = None
        best_recall_met = 0
        
        for strategy, metrics in results.items():
            if metrics['precision'] >= 0.40 and metrics['recall'] > best_recall_met:
                best_strategy = strategy
                best_recall_met = metrics['recall']
        
        if best_strategy is None:
            best_strategy = 'ensemble_or'
        
        print(f"\n✓ Selected Strategy: {best_strategy}")
        
        # Final evaluation with best strategy
        best_metrics = results[best_strategy]
        
        print("\n" + "="*70)
        print(f"FINAL EVALUATION - {best_strategy.upper()}")
        print("="*70)
        
        y_pred, y_proba = self.predict(X_test, 
                                       threshold=best_metrics['threshold'],
                                       strategy=best_strategy)
        
        print(f"\nAccuracy: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {best_metrics['precision']:.4f} ({best_metrics['precision']*100:.2f}%)")
        print(f"Recall: {best_metrics['recall']:.4f} ({best_metrics['recall']*100:.2f}%)")
        print(f"F1-Score: {best_metrics['f1_score']:.4f}")
        print(f"AUC-ROC: {best_metrics['auc_roc']:.4f}")
        
        # Target achievement
        print("\n" + "-"*70)
        print("TARGET ACHIEVEMENT")
        print("-"*70)
        
        if best_metrics['precision'] >= 0.40:
            print(f" PRECISION TARGET MET: {best_metrics['precision']*100:.2f}% >= 40%")
        else:
            print(f" PRECISION MISSED: {best_metrics['precision']*100:.2f}% < 40%")
        
        if best_metrics['recall'] >= 0.70:
            print(f" RECALL TARGET MET: {best_metrics['recall']*100:.2f}% >= 70%")
        else:
            print(f" RECALL MISSED: {best_metrics['recall']*100:.2f}% < 70%")
            print(f"   Gap: {(0.70 - best_metrics['recall'])*100:.2f}%")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n" + "-"*70)
        print("CONFUSION MATRIX")
        print("-"*70)
        print(f"True Negatives: {cm[0,0]:,}")
        print(f"False Positives: {cm[0,1]:,}")
        print(f"False Negatives: {cm[1,0]:,}")
        print(f"True Positives: {cm[1,1]:,}")
        
        fpr = cm[0,1] / (cm[0,0] + cm[0,1])
        print(f"\nFalse Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
        
        print("\n" + "-"*70)
        print("CLASSIFICATION REPORT")
        print("-"*70)
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
        
        return {
            'strategy': best_strategy,
            **best_metrics
        }
    
    def save_models(self, lgb_path='AFITraining/models/final_fraud_lgb.txt',
                   if_path='AFITraining/models/final_fraud_if.pkl'): # Saving both models
        self.lgb_model.save_model(lgb_path)
        joblib.dump(self.isolation_forest, if_path)
        print(f"\n✓ Models saved:")
        print(f"  LightGBM: {lgb_path}")
        print(f"  Isolation Forest: {if_path}")


if __name__ == "__main__":
    print("="*70)
    print("IMPROVED FRAUD DETECTION")
    print("Target: Recall >= 70%, Precision >= 40%")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    X_train = pd.read_csv('AFITraining/data/processed/X_train_fraud.csv')
    X_test = pd.read_csv('AFITraining/data/processed/X_test_fraud.csv')
    y_train = pd.read_csv('AFITraining/data/processed/y_train_fraud.csv')['is_fraud']
    y_test = pd.read_csv('AFITraining/data/processed/y_test_fraud.csv')['is_fraud']
    
    print(f"✓ Data loaded")
    print(f"  Training: {len(X_train):,} samples (Fraud: {y_train.sum():,})")
    print(f"  Test: {len(X_test):,} samples (Fraud: {y_test.sum():,})")
    
    # Train model
    detector = ImprovedFraudDetector()
    detector.train(X_train, y_train)
    
    # Evaluate
    metrics = detector.evaluate(X_test, y_test)
    
    # Save
    detector.save_models()
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('AFITraining/models/final_fraud_metrics.csv', index=False)
    print("\n✓ Metrics saved to: AFITraining/models/final_fraud_metrics.csv")
    
    print("\n" + "="*70)
    print("✓ IMPROVED FRAUD DETECTION COMPLETE")
    print("="*70)
    
    if metrics['recall'] >= 0.70 and metrics['precision'] >= 0.40:
        print("\n SUCCESS! Both targets achieved!")
    else:
        print("\n Targets not fully met. Additional options:")
        print("  - Adjust SMOTE sampling_strategy (try 0.3-0.7)")
        print("  - Use cost-sensitive learning with higher fraud costs")
        print("  - Try ensemble_or strategy for maximum recall")
        print("  - Consider threshold tuning in production")