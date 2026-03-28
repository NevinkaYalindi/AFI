# Integrated System - Unified Credit Scoring & Fraud Detection

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AFISystem:

    def __init__(self, use_improved_models=False):
        self.credit_model = None
        self.fraud_if_model = None
        self.fraud_lgb_model = None
        self.scaler = None
        self.feature_cols = None
        self.use_improved = use_improved_models
        
        # Dynamic thresholds for multi-level risk
        self.fraud_thresholds = {
            'low': -0.05,     
            'medium': -0.08,    
            'high': -0.10,     
            'critical': -0.10   
        }
        
        self.credit_thresholds = {
            'excellent': 80,   
            'good': 60,      
            'fair': 40,        
            'poor': 40          
        }
        
    def load_models(self,
               fraud_model_path='AFITraining/models/optimized_fraud_lgb_5M.txt',
               credit_model_path='AFITraining/models/optimized_credit_lgb_5M.txt',
               scaler_path='AFITraining/models/scaler_5M.pkl',
               feature_cols_path='AFITraining/models/feature_cols.json'):
    
        import json
        print("\n" + "="*60)
        print("LOADING AFI SYSTEM COMPONENTS")
        print("="*60)

        # Load fraud model
        print("\nLoading Fraud Detection Model...")
        self.fraud_lgb_model = lgb.Booster(model_file=fraud_model_path)
        print(f" Fraud model loaded: {fraud_model_path}")

        # Load credit model
        print("\nLoading Credit Scoring Model...")
        self.credit_model = lgb.Booster(model_file=credit_model_path)
        print(f" Credit model loaded: {credit_model_path}")

        # Load scaler
        print("\nLoading Scaler...")
        self.scaler = joblib.load(scaler_path)
        print(f" Scaler loaded: {scaler_path}")

        # Load feature columns
        with open(feature_cols_path, 'r') as f:
            self.feature_cols = json.load(f)
        print(f" Feature cols loaded: {len(self.feature_cols)} features")

        # Load thresholds from config
        import os
        if os.path.exists('AFITraining/models/fraud_config_5M.json'):
            with open('AFITraining/models/fraud_config_5M.json') as f:
                fraud_cfg = json.load(f)
                self.fraud_threshold = fraud_cfg.get('fraud_threshold', 0.30)
        else:
            self.fraud_threshold = 0.30

        if os.path.exists('AFITraining/models/credit_config_5M.json'):
            with open('AFITraining/models/credit_config_5M.json') as f:
                credit_cfg = json.load(f)
                self.credit_threshold = credit_cfg.get('credit_threshold', 0.30)
        else:
            self.credit_threshold = 0.30

        print("\n" + "="*60)
        print("AFI SYSTEM IS READY")
        print("="*60)
        
        
    def process_transaction(self, transaction_features):  # Process a single transaction through both models
    
        start_time = time.time()

        # Ensure correct feature order
        features_aligned = transaction_features[self.feature_cols]

        # Scale
        features_scaled = self.scaler.transform(features_aligned)
        features_scaled_df = pd.DataFrame(features_scaled, columns=self.feature_cols)

        # Fraud Detection
        fraud_proba = self.fraud_lgb_model.predict(features_scaled_df)[0]
        is_fraud    = int(fraud_proba > self.fraud_threshold)

        # Credit Scoring
        credit_risk_proba = self.credit_model.predict(features_scaled_df)[0]
        credit_score      = round((1 - credit_risk_proba) * 100, 2)

        # Fraud risk level
        if fraud_proba >= 0.80:
            fraud_risk_level = "CRITICAL"
        elif fraud_proba >= 0.60:
            fraud_risk_level = "HIGH"
        elif fraud_proba >= 0.30:
            fraud_risk_level = "MEDIUM"
        else:
            fraud_risk_level = "LOW"

        # Credit risk level
        if credit_score >= 71:
            credit_risk_level = "LOW RISK"
        elif credit_score >= 41:
            credit_risk_level = "MEDIUM RISK"
        else:
            credit_risk_level = "HIGH RISK"

        # Overall recommendation
        if fraud_risk_level in ["CRITICAL", "HIGH"]:
            overall_risk  = "HIGH"
            recommendation = f"REJECT — {fraud_risk_level} fraud risk detected"
        elif fraud_risk_level == "MEDIUM" or credit_score < 41:
            overall_risk  = "MEDIUM"
            recommendation = "REVIEW — Manual assessment required"
        else:
            overall_risk  = "LOW"
            recommendation = "APPROVE — Good creditworthiness, low fraud risk"

        processing_time = time.time() - start_time

        return {
            'credit_score':           credit_score,
            'credit_risk_level':      credit_risk_level,
            'credit_risk_probability':round(float(credit_risk_proba), 4),
            'fraud_probability':      round(float(fraud_proba), 4),
            'fraud_flag':             is_fraud,
            'fraud_risk_level':       fraud_risk_level,
            'overall_risk_level':     overall_risk,
            'recommendation':         recommendation,
            'processing_time_ms':     round(processing_time * 1000, 4),
            'timestamp':              datetime.now().isoformat(),
            'model_version':          '5M_kaggle'
    }
    
    def process_batch(self, transactions_df):
        # Process multiple transactions in batch
        print(f"\nProcessing {len(transactions_df):,} transactions...")
        start_time = time.time()

        # Ensuring correct feature order
        features_aligned = transactions_df[self.feature_cols]

        # Scale features
        features_scaled = self.scaler.transform(features_aligned)
        features_scaled_df = pd.DataFrame(features_scaled, columns=self.feature_cols)

        # Credit scoring
        credit_risk_proba = self.credit_model.predict(features_scaled_df)
        credit_scores = (1 - credit_risk_proba) * 100

        # Fraud detection — LightGBM only
        fraud_proba = self.fraud_lgb_model.predict(features_scaled_df)
        fraud_flags = (fraud_proba > self.fraud_threshold).astype(int)

        # Create results dataframe
        results = pd.DataFrame({
            'credit_score':            credit_scores.round(2),
            'credit_risk_probability': credit_risk_proba.round(4),
            'fraud_probability':       fraud_proba.round(4),
            'fraud_flag':              fraud_flags,
        })

        # Fraud risk level based on probability
        def get_fraud_risk_level(prob):
            if prob >= 0.80:
                return 'CRITICAL'
            elif prob >= 0.60:
                return 'HIGH'
            elif prob >= 0.30:
                return 'MEDIUM'
            else:
                return 'LOW'

        results['fraud_risk_level'] = results['fraud_probability'].apply(get_fraud_risk_level)

        # Credit risk level
        def get_credit_level(score):
            if score >= 71:
                return 'LOW RISK'
            elif score >= 41:
                return 'MEDIUM RISK'
            else:
                return 'HIGH RISK'

        results['credit_risk_level'] = results['credit_score'].apply(get_credit_level)

        # Overall risk and recommendation
        def get_overall_risk(row):
            if row['fraud_risk_level'] in ['CRITICAL', 'HIGH']:
                return 'HIGH', f"REJECT - {row['fraud_risk_level']} fraud risk"
            elif row['fraud_risk_level'] == 'MEDIUM' or row['credit_risk_level'] == 'HIGH RISK':
                return 'MEDIUM', 'REVIEW - Manual assessment required'
            elif row['credit_risk_level'] == 'MEDIUM RISK':
                return 'MEDIUM', 'REVIEW - Borderline creditworthiness'
            else:
                return 'LOW', 'APPROVE - Good creditworthiness'

        results[['overall_risk_level', 'recommendation']] = results.apply(
            lambda row: pd.Series(get_overall_risk(row)), axis=1
        )

        processing_time = time.time() - start_time
        avg_latency_ms = (processing_time / len(transactions_df)) * 1000

        print(f"  Batch processing complete")
        print(f"  Total time: {processing_time:.4f} seconds")
        print(f"  Average per transaction: {avg_latency_ms:.4f} ms")
        print(f"  Throughput: {len(transactions_df)/processing_time:,.0f} transactions")

        return results, avg_latency_ms
    
    def generate_report(self, results_df, y_true=None):
        # Summary report
        print("\n" + "="*60)
        print("AFI SYSTEM PROCESSING REPORT")
        print("="*60)
        
        print(f"\nTotal Transactions Processed: {len(results_df):,}")
        print(f"\nCredit Score Distribution:")
        print(f"  Mean: {results_df['credit_score'].mean():.2f}")
        print(f"  Median: {results_df['credit_score'].median():.2f}")
        print(f"  Std Dev: {results_df['credit_score'].std():.2f}")
        
        print(f"\nFraud Risk Level Distribution:")
        fraud_dist = results_df['fraud_risk_level'].value_counts()
        for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
            count = fraud_dist.get(level, 0)
            pct = (count / len(results_df)) * 100
            print(f"  {level}: {count:,} ({pct:.2f}%)")
        
        print(f"\nOverall Risk Level Distribution:")
        risk_dist = results_df['overall_risk_level'].value_counts()
        for level in [' LOW', ' MEDIUM', ' HIGH']:
            count = risk_dist.get(level, 0)
            pct = (count / len(results_df)) * 100
            print(f"  {level}: {count:,} ({pct:.2f}%)")
        
        print(f"\nRecommendation Distribution:")
        rec_dist = results_df['recommendation'].value_counts()
        for rec, count in rec_dist.head(5).items():
            pct = (count / len(results_df)) * 100
            print(f"  {rec}: {count:,} ({pct:.2f}%)")
        
        if y_true is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            fraud_accuracy = accuracy_score(y_true, results_df['fraud_flag'])
            fraud_precision = precision_score(y_true, results_df['fraud_flag'], zero_division=0)
            fraud_recall = recall_score(y_true, results_df['fraud_flag'], zero_division=0)
            fraud_f1 = f1_score(y_true, results_df['fraud_flag'], zero_division=0)
            
            print(f"\nIntegrated System Performance:")
            print(f"  Fraud Detection Accuracy: {fraud_accuracy:.4f}")
            print(f"  Fraud Detection Precision: {fraud_precision:.4f}")
            print(f"  Fraud Detection Recall: {fraud_recall:.4f}")
            print(f"  Fraud Detection F1-Score: {fraud_f1:.4f}")


if __name__ == "__main__":
    print("="*60)
    print("AFI - INTEGRATED SYSTEM DEMO")
    print("="*60)
    
    # Checking if improved models exist
    import os
    use_improved = (os.path.exists('AFITraining/models/improved_credit_scoring_model.txt') and 
                   os.path.exists('AFITraining/models/improved_fraud_if_model.pkl'))
    
    if use_improved:
        print("\n  Improved models detected - using enhanced version")
        print("  - SMOTE-balanced credit scoring")
        print("  - Ensemble fraud detection (IF + LightGBM)")
        print("  - Probability-based thresholds")
    else:
        print("\n  Using original models")
        print("  Run improve_credit_scoring.py and improve_fraud_detection.py first")
    
    # Initialize system
    afi = AFISystem(use_improved_models=use_improved)
    afi.load_models()
    
    # Load test data
    print("\nLoading test data for demonstration...")
    X_test = pd.read_csv('AFITraining/data/processed/X_test_credit_scoring.csv')
    y_test = pd.read_csv('AFITraining/data/processed/y_test_credit_scoring.csv')['is_fraud']
    
    # Single transaction
    print("\n" + "="*60)
    print("DEMO 1: SINGLE TRANSACTION PROCESSING")
    print("="*60)
    
    single_transaction = X_test.iloc[[0]]
    result = afi.process_transaction(single_transaction)
    
    print("\nTransaction Analysis:")
    for key, value in result.items():
        if value is not None:
            print(f"  {key}: {value}")
    
    # Batch processing
    print("\n" + "="*60)
    print("DEMO 2: BATCH PROCESSING")
    print("="*60)
    
    batch_transactions = X_test.head(1000)
    batch_y_true = y_test.head(1000)
    
    results_df, avg_latency = afi.process_batch(batch_transactions)
    
    # Generate report
    afi.generate_report(results_df, batch_y_true)
    
    # NFR3 check
    print("\n" + "="*60)
    print("NFR3 COMPLIANCE CHECK")
    print("="*60)
    if avg_latency < 1000:
        print(f" MEETS NFR3: {avg_latency:.4f} ms < 1000 ms")
    
    # Fraud flagging analysis
    print("\n" + "="*60)
    print("FRAUD FLAGGING ANALYSIS")
    print("="*60)
    
    flagged_rate = (results_df['fraud_flag'].sum() / len(results_df)) * 100
    high_risk_rate = (results_df['overall_risk_level'] == ' HIGH').sum() / len(results_df) * 100
    
    print(f"Fraud flags raised: {results_df['fraud_flag'].sum():,} ({flagged_rate:.2f}%)")
    print(f"HIGH risk assessments: {(results_df['overall_risk_level'] == ' HIGH').sum():,} ({high_risk_rate:.2f}%)")
    
    if use_improved:
        if high_risk_rate < 10:
            print(f" TARGET MET: {high_risk_rate:.1f}% < 10%")
    
    # Save results
    results_df.to_csv('AFITraining/models/improved_demo_results.csv', index=False)
    print(f"\n✓ Demo results saved to: AFITraining/models/improved_demo_results.csv")
    
    print("\n" + "="*60)
    print(" INTEGRATED SYSTEM DEMO COMPLETE")
    print("="*60)
    
    if use_improved:
        print("\n SUCCESS! Improved system deployed:")
        print(f"  - Multi-level risk assessment")
        print(f"  - Latency: {avg_latency:.4f} ms")