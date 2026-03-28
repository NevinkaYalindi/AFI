# Improved Fraud Detection with Ensemble Approach

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report)
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

class ImprovedFraudDetectionModel:
    
    def __init__(self, contamination=0.02):
        # Initialize ensemble fraud detection system
        
        # Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=150,  
            max_samples='auto',
            max_features=0.8,  
            bootstrap=False,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        # LightGBM
        self.lgb_model = None
        self.contamination = contamination
        
    def train(self, X_train_normal, X_train_full, y_train_full):

        # Training ensemble model
        print("\n" + "="*60)
        print("IMPROVED FRAUD DETECTION MODEL TRAINING")
        print("="*60)
        
        # Training Isolation Forest
        print(f"\n[1/2] Training Isolation Forest (Unsupervised)")
        print(f"  Training on normal transactions only")
        print(f"  Normal samples: {len(X_train_normal):,}")
        print(f"  Contamination: {self.contamination*100:.2f}%")
        
        start_time = time.time()
        self.isolation_forest.fit(X_train_normal)
        if_time = time.time() - start_time
        
        print(f"   Isolation Forest trained in {if_time:.2f} seconds")
        
        # Training LightGBM
        print(f"\n[2/2] Training LightGBM Fraud Classifier (Supervised)")
        print(f"  Training samples: {len(X_train_full):,}")
        print(f"  Fraud ratio: {y_train_full.mean()*100:.2f}%")
        
        # LightGBM parameters optimized for fraud detection
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 6,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'scale_pos_weight': (len(y_train_full) - y_train_full.sum()) / y_train_full.sum(),
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        train_data = lgb.Dataset(X_train_full, label=y_train_full)
        
        start_time = time.time()
        self.lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            callbacks=[lgb.log_evaluation(period=100)]
        )
        lgb_time = time.time() - start_time
        
        print(f"   LightGBM trained in {lgb_time:.2f} seconds")
        
        print(f"\n Ensemble model training complete")
        print(f"  Total training time: {if_time + lgb_time:.2f} seconds")
        
    def predict_ensemble(self, X, method='voting'):

        # Isolation Forest predictions
        if_predictions = self.isolation_forest.predict(X)
        if_anomaly_scores = self.isolation_forest.decision_function(X)
        if_fraud_flags = np.where(if_predictions == -1, 1, 0)
        
        # LightGBM predictions
        lgb_proba = self.lgb_model.predict(X)
        lgb_fraud_flags = (lgb_proba >= 0.5).astype(int)
        
        if method == 'voting':
            ensemble_fraud = np.where((if_fraud_flags == 1) & (lgb_fraud_flags == 1), 1, 0)
        elif method == 'if_only':
            ensemble_fraud = if_fraud_flags
        elif method == 'lgb_only':
            ensemble_fraud = lgb_fraud_flags
        else:
            ensemble_fraud = if_fraud_flags  
        
        return ensemble_fraud, if_anomaly_scores, lgb_proba
        
    def evaluate(self, X_test, y_test): # Evaluate ensemble fraud detection performance
      
        print("\n" + "="*60)
        print("IMPROVED FRAUD DETECTION EVALUATION")
        print("="*60)
        
        # Generate predictions
        print("\nGenerating ensemble predictions...")
        start_time = time.time()
        
        ensemble_pred, if_scores, lgb_proba = self.predict_ensemble(X_test, method='voting')
        
        prediction_time = time.time() - start_time
        
        # getting individual model predictions for comparison
        if_pred_only, _, _ = self.predict_ensemble(X_test, method='if_only')
        lgb_pred_only, _, _ = self.predict_ensemble(X_test, method='lgb_only')
        
        # Calculate metrics for ensemble
        accuracy = accuracy_score(y_test, ensemble_pred)
        precision = precision_score(y_test, ensemble_pred, zero_division=0)
        recall = recall_score(y_test, ensemble_pred, zero_division=0)
        f1 = f1_score(y_test, ensemble_pred, zero_division=0)
        auc = roc_auc_score(y_test, lgb_proba)
        
        # Individual model metrics
        if_precision = precision_score(y_test, if_pred_only, zero_division=0)
        if_recall = recall_score(y_test, if_pred_only, zero_division=0)
        
        lgb_precision = precision_score(y_test, lgb_pred_only, zero_division=0)
        lgb_recall = recall_score(y_test, lgb_pred_only, zero_division=0)
        
        # Latency
        avg_latency_ms = (prediction_time / len(X_test)) * 1000
        
        print("\n" + "-"*60)
        print("ENSEMBLE PERFORMANCE (Voting: Both models agree)")
        print("-"*60)
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        
        # Check targets
        print("\n" + "-"*60)
        print("TARGET ACHIEVEMENT")
        print("-"*60)
        
        if precision >= 0.40:
            print(f" PRECISION TARGET MET: {precision*100:.2f}% >= 40%")
        else:
            print(f" PRECISION TARGET MISSED: {precision*100:.2f}% < 40%")
            
        if recall >= 0.70:
            print(f" RECALL TARGET MET: {recall*100:.2f}% >= 70%")
        else:
            print(f" RECALL BELOW TARGET: {recall*100:.2f}% < 70%")
        
        # Improvement comparison
        print("\n" + "-"*60)
        print("IMPROVEMENT OVER ORIGINAL ISOLATION FOREST")
        print("-"*60)
        print(f"Precision: 23.24% → {precision*100:.2f}% ({(precision-0.2324)*100:+.2f}%)")
        print(f"Recall:    29.13% → {recall*100:.2f}% ({(recall-0.2913)*100:+.2f}%)")
        
        # Model comparison
        print("\n" + "-"*60)
        print("INDIVIDUAL MODEL COMPARISON")
        print("-"*60)
        print(f"{'Model':<25} {'Precision':<12} {'Recall':<12}")
        print("-" * 60)
        print(f"{'Isolation Forest Only':<25} {if_precision*100:>10.2f}%  {if_recall*100:>10.2f}%")
        print(f"{'LightGBM Only':<25} {lgb_precision*100:>10.2f}%  {lgb_recall*100:>10.2f}%")
        print(f"{'Ensemble (Voting)':<25} {precision*100:>10.2f}%  {recall*100:>10.2f}%")
        
        # Latency
        print("\n" + "-"*60)
        print("LATENCY ANALYSIS")
        print("-"*60)
        print(f"Total prediction time: {prediction_time:.4f} seconds")
        print(f"Average per transaction: {avg_latency_ms:.4f} ms")
        print(f"Predictions per second: {len(X_test)/prediction_time:,.0f}")
        
        if avg_latency_ms < 1000:
            print(f" MEETS NFR3: {avg_latency_ms:.4f} ms < 1000 ms")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, ensemble_pred)
        print("\n" + "-"*60)
        print("CONFUSION MATRIX (Ensemble)")
        print("-"*60)
        print(f"True Negatives:  {cm[0,0]:,} ") # Correctly identified normal
        print(f"False Positives: {cm[0,1]:,} ") # Normal flagged as fraud
        print(f"False Negatives: {cm[1,0]:,} ") # Fraud missed
        print(f"True Positives:  {cm[1,1]:,} ") # Correctly detected fraud
        
        fpr = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        print(f"\nFalse Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
        
        # Classification Report
        print("\n" + "-"*60)
        print("CLASSIFICATION REPORT (Ensemble)")
        print("-"*60)
        print(classification_report(y_test, ensemble_pred, target_names=['Normal (0)', 'Fraud (1)'], zero_division=0))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'false_positive_rate': fpr,
            'avg_latency_ms': avg_latency_ms,
            'predictions_per_second': len(X_test)/prediction_time,
            'improvement_precision': (precision - 0.2324) * 100,
            'improvement_recall': (recall - 0.2913) * 100,
            'if_precision': if_precision,
            'if_recall': if_recall,
            'lgb_precision': lgb_precision,
            'lgb_recall': lgb_recall
        }
    
    def save_models(self, if_path='AFITraining/models/improved_fraud_if_model.pkl', lgb_path='AFITraining/models/improved_fraud_lgb_model.txt'):
        # Saving both ensemble models
        joblib.dump(self.isolation_forest, if_path)
        self.lgb_model.save_model(lgb_path)
        print(f"\n Isolation Forest saved to: {if_path}")
        print(f" LightGBM saved to: {lgb_path}")


if __name__ == "__main__":
    print("="*60)
    print("FRAUD DETECTION IMPROVEMENT WITH ENSEMBLE")
    print("="*60)
    
    print("\nLoading preprocessed data...")
    
    # Load fraud detection data
    X_train_normal = pd.read_csv('AFITraining/data/processed/X_train_normal_fraud.csv')
    X_train_full = pd.read_csv('AFITraining/data/processed/X_train_fraud.csv')
    y_train_full = pd.read_csv('AFITraining/data/processed/y_train_fraud.csv')['is_fraud']
    X_test = pd.read_csv('AFITraining/data/processed/X_test_fraud.csv')
    y_test = pd.read_csv('AFITraining/data/processed/y_test_fraud.csv')['is_fraud']
    
    print(f" Data loaded successfully")
    print(f"  Normal training samples: {len(X_train_normal):,}")
    print(f"  Full training samples: {len(X_train_full):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Fraud in test: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")
    
    # Initialize and train ensemble model
    model = ImprovedFraudDetectionModel(contamination=0.02)
    model.train(X_train_normal, X_train_full, y_train_full)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Save models
    import os
    os.makedirs('AFITraining/models', exist_ok=True)
    model.save_models()
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('AFITraining/models/improved_fraud_detection_metrics.csv', index=False)
    print("\n Metrics saved to: AFITraining/models/improved_fraud_detection_metrics.csv")
    
    # Summary
    print("\n" + "="*60)
    print(" FRAUD DETECTION IMPROVEMENT COMPLETE")
    print("="*60)
    
    if metrics['precision'] >= 0.40:
        print(f"\n PRECISION TARGET ACHIEVED: {metrics['precision']*100:.2f}% >= 40% ")
    else:
        print(f"\n  Precision: {metrics['precision']*100:.2f}% < 40%")
        print("    Consider: Adjusting contamination or voting threshold")
    
    if metrics['recall'] >= 0.70:
        print(f" RECALL TARGET ACHIEVED: {metrics['recall']*100:.2f}% >= 70% ")
    else:
        print(f"  Recall: {metrics['recall']*100:.2f}% < 70%")
        print("   Trade-off: Higher precision often reduces recall")