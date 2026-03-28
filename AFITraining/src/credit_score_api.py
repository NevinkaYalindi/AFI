
# Old 5M model

"""
AFI - Real-Time Credit Score Prediction API
Purpose: Predict credit scores for new loan applicants using transaction history
Validates: FR03 - "Compute credit scores for both banked and unbanked users based on alternative data"
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CreditScorePredictor:
    """
    Real-time credit score prediction for loan applicants
    Based on their transaction history (alternative data)
    """
    
    def __init__(self, 
             model_path='AFITraining/models/optimized_credit_lgb_5M.txt',
             scaler_path='AFITraining/models/scaler_5M.pkl'):

        print("Initializing AFI Credit Score Predictor...") # Initialize the predictor with trained model and scaler
        
        try:
            self.model = joblib.load(model_path)
            print(f" Loaded tuned model: {model_path}")
        except:
            try:
                model_path = 'AFITraining/models/improved_credit_scoring_model.txt'
                self.model = lgb.Booster(model_file=model_path)
                print(f" Loaded improved model: {model_path}")
            except:
                model_path = 'AFITraining/models/credit_scoring_model.txt'
                self.model = lgb.Booster(model_file=model_path)
                print(f" Loaded original model: {model_path}")
        
        self.scaler = joblib.load(scaler_path)
        print(f" Loaded scaler: {scaler_path}")
        
       # Loading exact feature names from trained model (29 features)
        import json
        try:
            with open('models/feature_cols.json', 'r') as f:
                self.feature_names = json.load(f)
            print(f"✓ Loaded {len(self.feature_names)} features from feature_cols.json")
        except FileNotFoundError:
            # Fallback — hardcoded 29 features from Kaggle training
            self.feature_names = [
                'amount', 'transaction_type', 'merchant_category', 'device_used',
                'fraud_type', 'time_since_last_transaction', 'spending_deviation_score',
                'velocity_score', 'geo_anomaly_score', 'payment_channel',
                'hour', 'day_of_week', 'is_weekend', 'is_night', 'month',
                'log_amount', 'is_round_amount', 'avg_amount', 'std_amount',
                'max_amount', 'total_amount', 'fraud_history_ratio',
                'spending_consistency', 'activity_score', 'amount_deviation',
                'is_large_tx', 'recv_tx_count', 'recv_fraud_cnt',
                'receiver_risk_score'
            ]
            print(f" feature_cols.json not found")

        print(" Predictor ready!\n")
    
    def engineer_features_from_transactions(self, transactions_df):
        """
        Engineer features from raw transaction data
        This simulates the feature engineering pipeline for new applicants
        """
        print(f"Engineering features from {len(transactions_df)} transactions...")
        
        df = transactions_df.copy()
        
        # Time-based features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour'].isin(range(0, 6)).astype(int)
        
        # Sender behavioral features (credit scoring focus)
        sender_stats = df.groupby('sender_account').agg({
            'amount': ['count', 'mean', 'std', 'min', 'max', 'sum'],
            'transaction_type': lambda x: x.nunique(),
            'device_used': lambda x: x.nunique(),
        }).reset_index()
        
        sender_stats.columns = ['sender_account', 'tx_count', 'avg_amount',
                                'std_amount', 'min_amount', 'max_amount',
                                'total_spent', 'tx_type_variety', 'device_variety']
        
        # Handling NaN in std_amount (happens when only 1 transaction)
        sender_stats['std_amount'] = sender_stats['std_amount'].fillna(0)
        
        # Behavioral scores
        sender_stats['spending_consistency'] = 1 - (sender_stats['std_amount'] / 
                                                    (sender_stats['avg_amount'] + 1))
        sender_stats['activity_score'] = np.log1p(sender_stats['tx_count'])
        
        # Assume no fraud history for new applicants
        sender_stats['fraud_count'] = 0
        sender_stats['fraud_history_ratio'] = 0.0
        sender_stats['location_variety'] = 1  # Default value
        
        # Merge back
        df = df.merge(sender_stats, on='sender_account', how='left')
        
        # Ensure avg_amount exists and has no NaN values
        if 'avg_amount' not in df.columns:
            # If merge failed, calculate directly
            df['avg_amount'] = df.groupby('sender_account')['amount'].transform('mean')
        else:
            # Fill any NaN values
            df['avg_amount'] = df['avg_amount'].fillna(df['amount'])
        
        # Also ensure other stats columns exist
        for col in ['std_amount', 'min_amount', 'max_amount', 'total_spent', 
                    'tx_count', 'tx_type_variety', 'device_variety',
                    'spending_consistency', 'activity_score']:
            if col not in df.columns:
                if col == 'tx_count':
                    df[col] = len(df)
                elif col == 'total_spent':
                    df[col] = df['amount'].sum()
                elif col in ['min_amount', 'max_amount', 'std_amount']:
                    df[col] = df['amount']
                else:
                    df[col] = 1
        
        # Receiver features (simplified for new applicants)
        df['receiver_tx_count'] = 1
        df['receiver_avg_amount'] = df['amount']
        df['receiver_fraud_count'] = 0
        df['receiver_risk_score'] = 0.0
        
        # Velocity features
        df = df.sort_values(['sender_account', 'timestamp'])
        df['time_since_last_tx_hours'] = df.groupby('sender_account')['timestamp'].diff().dt.total_seconds() / 3600
        df['time_since_last_tx_hours'] = df['time_since_last_tx_hours'].fillna(999)
        df['is_rapid_transaction'] = (df['time_since_last_tx_hours'] < 0.1).astype(int)
        
        # Amount-based features
        df['amount_deviation'] = abs(df['amount'] - df['avg_amount']) / (df['avg_amount'] + 1)
        df['is_large_tx'] = (df['amount'] > df['avg_amount'] * 3).astype(int)
        df['is_round_amount'] = (df['amount'] % 100 == 0).astype(int)
        
        # Categorical encoding
        tx_type_map = {'deposit': 1, 'withdrawal': 2, 'transfer': 3, 'payment': 4}
        df['tx_type_encoded'] = df['transaction_type'].map(tx_type_map).fillna(0)
        
        device_map = {'mobile': 1, 'web': 2, 'atm': 3, 'pos': 4}
        df['device_encoded'] = df['device_used'].map(device_map).fillna(0)
        
        if 'merchant_category' in df.columns:
            df['merchant_encoded'] = pd.Categorical(df['merchant_category']).codes
        else:
            df['merchant_encoded'] = 0
        
        # Fill missing values
        for col in ['time_since_last_transaction', 'spending_deviation_score',
                    'velocity_score', 'geo_anomaly_score', 'fraud_count',
                    'fraud_history_ratio', 'location_variety']:
            if col not in df.columns:
                df[col] = 0
        
        print("✓ Feature engineering complete")
        return df
    
    def predict_credit_score(self, applicant_transactions, applicant_id=None):
        """
        Predict credit score for a loan applicant based on their transactions
        """
        start_time = datetime.now()
        
        # Engineer features
        features_df = self.engineer_features_from_transactions(applicant_transactions)
        
        # Getting the latest transaction features
        latest_features = features_df[self.feature_names].iloc[-1:].copy()
        
        # Scale features
        features_scaled = self.scaler.transform(latest_features)
        features_scaled_df = pd.DataFrame(features_scaled, columns=self.feature_names)
        
        # Predict
        if hasattr(self.model, 'predict_proba'):
            # Sklearn-based model
            fraud_risk_proba = self.model.predict_proba(features_scaled_df)[0][1]
        else:
            # LightGBM booster
            fraud_risk_proba = self.model.predict(features_scaled_df)[0]
        
        # Convert fraud probability to credit score
        credit_score = (1 - fraud_risk_proba) * 100
        
        # Determine credit risk level
        if credit_score >= 80:
            risk_level = "EXCELLENT"
            risk_category = " LOW RISK"
            recommendation = "APPROVE - Excellent creditworthiness"
        elif credit_score >= 60:
            risk_level = "GOOD"
            risk_category = " LOW RISK"
            recommendation = "APPROVE - Good creditworthiness"
        elif credit_score >= 40:
            risk_level = "FAIR"
            risk_category = " MEDIUM RISK"
            recommendation = "REVIEW - Borderline creditworthiness, manual assessment recommended"
        else:
            risk_level = "POOR"
            risk_category = " HIGH RISK"
            recommendation = "REJECT - High credit risk detected"
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Compile result
        result = {
            'applicant_id': applicant_id or applicant_transactions['sender_account'].iloc[0],
            'credit_score': round(credit_score, 2),
            'fraud_risk_probability': round(fraud_risk_proba, 4),
            'risk_level': risk_level,
            'risk_category': risk_category,
            'recommendation': recommendation,
            'transaction_history_days': len(applicant_transactions),
            'total_transaction_volume': float(applicant_transactions['amount'].sum()),
            'avg_transaction_amount': float(applicant_transactions['amount'].mean()),
            'processing_time_ms': round(processing_time, 2),
            'timestamp': datetime.now().isoformat(),
            'model_version': 'tuned' if 'tuned' in str(self.model) else 'production'
        }
        
        return result
    
    def display_credit_assessment(self, result):
        # Display credit assessment in formatted output
        print("\n" + "="*70)
        print("AFI CREDIT SCORE ASSESSMENT")
        print("="*70)
        print(f"\nApplicant ID: {result['applicant_id']}")
        print(f"Assessment Time: {result['timestamp']}")
        
        print("\n" + "-"*70)
        print("CREDIT SCORE ANALYSIS")
        print("-"*70)
        print(f"Credit Score: {result['credit_score']:.2f}/100")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Risk Category: {result['risk_category']}")
        print(f"Fraud Risk: {result['fraud_risk_probability']*100:.2f}%")
        
        print("\n" + "-"*70)
        print("TRANSACTION PROFILE")
        print("-"*70)
        print(f"Transaction History: {result['transaction_history_days']} transactions")
        print(f"Total Volume: ${result['total_transaction_volume']:,.2f}")
        print(f"Average Amount: ${result['avg_transaction_amount']:,.2f}")
        
        print("\n" + "-"*70)
        print("RECOMMENDATION")
        print("-"*70)
        print(f"Decision: {result['recommendation']}")
        
        print("\n" + "-"*70)
        print("SYSTEM PERFORMANCE")
        print("-"*70)
        print(f"Processing Time: {result['processing_time_ms']:.2f} ms")
        print(f"Model Version: {result['model_version']}")
        
        print("\n" + "="*70)


def demo_single_applicant():
    # Assessing a single loan applicant based on their transaction history
    
    print("="*70)
    print("DEMO: SINGLE LOAN APPLICANT CREDIT ASSESSMENT")
    print("="*70)
    
    # Load sample transaction data
    print("\nLoading sample transactions...")
    all_transactions = pd.read_csv('AFITraining/data/processed/financial_transactions_engineered.csv')
    
    # Select a random applicant
    random_applicant = all_transactions['sender_account'].sample(1).iloc[0]
    applicant_transactions = all_transactions[
        all_transactions['sender_account'] == random_applicant
    ].head(50)  # Last 50 transactions
    
    print(f" Loaded {len(applicant_transactions)} transactions for applicant: {random_applicant}")
    
    # Initialize predictor
    predictor = CreditScorePredictor()
    
    # Predict credit score
    result = predictor.predict_credit_score(applicant_transactions, applicant_id=random_applicant)
    
    # Display result
    predictor.display_credit_assessment(result)
    
    return result


def demo_batch_applicants(n_applicants=10):
    # Assessing multiple loan applicants in batch

    print("="*70)
    print(f"DEMO: BATCH CREDIT ASSESSMENT ({n_applicants} APPLICANTS)")
    print("="*70)
    
    # Load sample data
    print("\nLoading sample transactions...")
    all_transactions = pd.read_csv('AFITraining/data/processed/financial_transactions_engineered.csv')
    
    # Select random applicants
    applicants = all_transactions['sender_account'].sample(n_applicants).unique()
    
    # Initialize predictor
    predictor = CreditScorePredictor()
    
    # Process each applicant
    results = []
    for i, applicant_id in enumerate(applicants, 1):
        print(f"\n[{i}/{n_applicants}] Processing applicant: {applicant_id}")
        
        applicant_transactions = all_transactions[
            all_transactions['sender_account'] == applicant_id
        ].head(30)
        
        result = predictor.predict_credit_score(applicant_transactions, applicant_id=applicant_id)
        results.append(result)
        
        print(f"  Credit Score: {result['credit_score']:.2f} | Risk: {result['risk_level']}")
    
    # Summary statistics
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("BATCH ASSESSMENT SUMMARY")
    print("="*70)
    
    print(f"\nTotal Applicants Processed: {len(results_df)}")
    print(f"Average Credit Score: {results_df['credit_score'].mean():.2f}")
    print(f"Average Processing Time: {results_df['processing_time_ms'].mean():.2f} ms")
    
    print("\nRisk Level Distribution:")
    risk_dist = results_df['risk_level'].value_counts()
    for risk, count in risk_dist.items():
        pct = (count / len(results_df)) * 100
        print(f"  {risk}: {count} ({pct:.1f}%)")
    
    print("\nRecommendation Distribution:")
    rec_dist = results_df['recommendation'].value_counts()
    for rec, count in rec_dist.head(3).items():
        pct = (count / len(results_df)) * 100
        print(f"  {rec}: {count} ({pct:.1f}%)")
    
    # Save results
    results_df.to_csv('AFITraining/models/loan_applicants_assessment.csv', index=False)
    print("\n✓ Results saved to: AFITraining/models/loan_applicants_assessment.csv")
    
    return results_df


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("AFI - LOAN APPLICANT CREDIT SCORE PREDICTION")
    print("Compute credit scores based on alternative data")
    print("="*70)
    
    # Check command line argument
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        # Batch mode
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        results = demo_batch_applicants(n_applicants=n)
    else:
        # Single applicant mode
        result = demo_single_applicant()
    
    print("\n" + "="*70)
    print("CREDIT SCORE PREDICTION COMPLETE")
    print("="*70)
    
    print("\nUsage:")
    print("  Single applicant: python credit_score_api.py")
    print("  Batch mode: python credit_score_api.py --batch 20")