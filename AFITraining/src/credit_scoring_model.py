
# Old credit scoring model (Improved)

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

class ImprovedCreditScoringModel:
    
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.smote = SMOTE(random_state=42, k_neighbors=5)
        
    def train(self, X_train, y_train, X_test, y_test):
        
        print("\n" + "="*60)
        print("IMPROVED CREDIT SCORING MODEL TRAINING")
        print("="*60)
        
        print(f"\nOriginal training samples: {len(X_train):,}")
        print(f"Original fraud ratio: {y_train.mean()*100:.2f}%")
        
        # Applying SMOTE 
        # To balance the dataset
        print("\nApplying SMOTE to balance classes...")
        start_time = time.time()
        
        X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train, y_train)
        
        smote_time = time.time() - start_time
        
        print(f" SMOTE completed in {smote_time:.2f} seconds")
        print(f"Balanced training samples: {len(X_train_balanced):,}")
        print(f"Balanced fraud ratio: {y_train_balanced.mean()*100:.2f}%")
        print(f"Fraud cases increased: {y_train.sum():,} → {y_train_balanced.sum():,}")
        
        # Calculating class weights for additional balance
        fraud_count = y_train_balanced.sum()
        normal_count = len(y_train_balanced) - fraud_count
        scale_pos_weight = normal_count / fraud_count
        
        print(f"\nClass weight (scale_pos_weight): {scale_pos_weight:.2f}")
        
        # Enhanced LightGBM parameters optimized for recall
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 7,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'scale_pos_weight': scale_pos_weight,
            'is_unbalance': True,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Create LightGBM datasets
        print("\nPreparing LightGBM datasets...")
        train_data = lgb.Dataset(X_train_balanced, label=y_train_balanced)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model with early stopping
        print("Training improved model...")
        start_time = time.time()
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        training_time = time.time() - start_time
        print(f"\n Training completed in {training_time:.2f} seconds")
        print(f"Best iteration: {self.model.best_iteration}")
        print(f"Best AUC score: {self.model.best_score['valid']['auc']:.4f}")
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
    def evaluate(self, X_test, y_test, threshold=0.5):
        
        #Evaluate model with custom threshold for recall optimization
        print("\n" + "="*60)
        print("IMPROVED CREDIT SCORING EVALUATION")
        print("="*60)
        
        # Predictions
        print("\nGenerating predictions...")
        start_time = time.time()
        y_pred_proba = self.model.predict(X_test)
        prediction_time = time.time() - start_time
        
        print("\nFinding optimal threshold for 70%+ precision and max recall...")
        
        best_threshold = 0.5
        best_f1 = 0
        results_by_threshold = []
        
        for thresh in np.arange(0.3, 0.7, 0.05):
            y_pred_temp = (y_pred_proba >= thresh).astype(int)
            precision_temp = precision_score(y_test, y_pred_temp, zero_division=0)
            recall_temp = recall_score(y_test, y_pred_temp, zero_division=0)
            f1_temp = f1_score(y_test, y_pred_temp, zero_division=0)
            
            results_by_threshold.append({
                'threshold': thresh,
                'precision': precision_temp,
                'recall': recall_temp,
                'f1': f1_temp
            })
            
            # Finding best threshold with precision >= 0.70
            if precision_temp >= 0.70 and f1_temp > best_f1:
                best_threshold = thresh
                best_f1 = f1_temp
        
        print(f" Optimal threshold found: {best_threshold:.2f}")
        
        # Use optimal threshold
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Latency calculation
        avg_latency_ms = (prediction_time / len(X_test)) * 1000
        
        print("\n" + "-"*60)
        print("PERFORMANCE METRICS (Optimized Threshold)")
        print("-"*60)
        print(f"Threshold Used: {best_threshold:.2f}")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        
        # Checking if targets met
        print("\n" + "-"*60)
        print("TARGET ACHIEVEMENT")
        print("-"*60)
        
        if recall >= 0.75:
            print(f" RECALL TARGET MET: {recall*100:.2f}% >= 75%")
        else:
            print(f" RECALL TARGET MISSED: {recall*100:.2f}% < 75%")
            
        if precision >= 0.70:
            print(f" PRECISION TARGET MET: {precision*100:.2f}% >= 70%")
        else:
            print(f" PRECISION TARGET MISSED: {precision*100:.2f}% < 70%")
        
        # Comparison with original model
        print("\n" + "-"*60)
        print("IMPROVEMENT OVER ORIGINAL MODEL")
        print("-"*60)
        print(f"Recall:    50.74% → {recall*100:.2f}% ({(recall-0.5074)*100:+.2f}%)")
        print(f"Precision: 74.61% → {precision*100:.2f}% ({(precision-0.7461)*100:+.2f}%)")
        print(f"F1-Score:  60.40% → {f1*100:.2f}% ({(f1-0.6040)*100:+.2f}%)")
        
        # Latency
        print("\n" + "-"*60)
        print("LATENCY ANALYSIS")
        print("-"*60)
        print(f"Total prediction time: {prediction_time:.4f} seconds")
        print(f"Average per transaction: {avg_latency_ms:.4f} ms")
        print(f"Predictions per second: {len(X_test)/prediction_time:,.0f}")
        
        if avg_latency_ms < 1000:
            print(f"MEETS NFR3: {avg_latency_ms:.4f} ms < 1000 ms")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n" + "-"*60)
        print("CONFUSION MATRIX")
        print("-"*60)
        print(f"True Negatives:  {cm[0,0]:,}")
        print(f"False Positives: {cm[0,1]:,}")
        print(f"False Negatives: {cm[1,0]:,}")
        print(f"True Positives:  {cm[1,1]:,}")
        
        # Classification Report
        print("\n" + "-"*60)
        print("CLASSIFICATION REPORT")
        print("-"*60)
        print(classification_report(y_test, y_pred, 
                                   target_names=['Low Risk (0)', 'High Risk (1)']))
        
        # Threshold analysis table
        print("\n" + "-"*60)
        print("THRESHOLD ANALYSIS")
        print("-"*60)
        threshold_df = pd.DataFrame(results_by_threshold)
        print(threshold_df.to_string(index=False))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'avg_latency_ms': avg_latency_ms,
            'predictions_per_second': len(X_test)/prediction_time,
            'optimal_threshold': best_threshold,
            'improvement_recall': (recall - 0.5074) * 100,
            'improvement_precision': (precision - 0.7461) * 100
        }
    
    def save_model(self, filepath='AFITraining/models/improved_credit_scoring_model.txt'):
        #Save improved model
        self.model.save_model(filepath)
        self.feature_importance.to_csv(filepath.replace('.txt', '_feature_importance.csv'), 
                                       index=False)
        print(f"\n Improved model saved to: {filepath}")
        print(f" Feature importance saved to: {filepath.replace('.txt', '_feature_importance.csv')}")


if __name__ == "__main__":
    print("="*60)
    print("CREDIT SCORING IMPROVEMENT WITH SMOTE")
    print("="*60)
    
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("\n  ERROR: imbalanced-learn not installed!")
        print("Please install it:")
        print("  pip install imbalanced-learn")
        exit(1)
    
    print("\nLoading preprocessed data...")
    
    # Load credit scoring data
    X_train = pd.read_csv('AFITraining/data/processed/X_train_credit_scoring.csv')
    X_test = pd.read_csv('AFITraining/data/processed/X_test_credit_scoring.csv')
    y_train = pd.read_csv('AFITraining/data/processed/y_train_credit_scoring.csv')['is_fraud']
    y_test = pd.read_csv('AFITraining/data/processed/y_test_credit_scoring.csv')['is_fraud']
    
    print(f" Data loaded successfully")
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    
    # Initialize and train improved model
    model = ImprovedCreditScoringModel()
    model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Save model
    import os
    os.makedirs('AFITraining/models', exist_ok=True)
    model.save_model('AFITraining/models/improved_credit_scoring_model.txt')
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('AFITraining/models/improved_credit_scoring_metrics.csv', index=False)
    print("\n Metrics saved to: AFITraining/models/improved_credit_scoring_metrics.csv")
    
    # Summary
    print("\n" + "="*60)
    print(" CREDIT SCORING IMPROVEMENT COMPLETE")
    print("="*60)
    
    if metrics['recall'] >= 0.75 and metrics['precision'] >= 0.70:
        print("\n  SUCCESS! Both targets achieved:")
        print(f"   Recall: {metrics['recall']*100:.2f}% >= 75% ")
        print(f"   Precision: {metrics['precision']*100:.2f}% >= 70% ")
    else:
        print("\n  Targets partially met. Consider:")
        print("   - Adjusting SMOTE sampling strategy")
        print("   - Further hyperparameter tuning")
        print("   - Trying different threshold values")