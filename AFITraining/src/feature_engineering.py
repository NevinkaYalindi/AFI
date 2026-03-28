# Feature Engineering for Alternative Data

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    #Generating alternative data features from transaction records
    
    def __init__(self, df):
        self.df = df.copy()
        
    def parse_timestamps(self):
        #Extracting time-based features
        print("Extracting time-based features...")

        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='mixed')
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        self.df['is_night'] = self.df['hour'].isin(range(0, 6)).astype(int)
        print(" Time features created")
        
    def create_sender_behavioral_features(self):
        # Creating behavioral features per sender
        print("Creating sender behavioral features...")
        
        # Converting is_fraud to int for aggregation
        self.df['is_fraud_int'] = self.df['is_fraud'].astype(int)
        
        # Transaction frequency and patterns
        sender_stats = self.df.groupby('sender_account').agg({
            'transaction_id': 'count',
            'amount': ['mean', 'std', 'min', 'max', 'sum'],
            'is_fraud_int': 'sum',
            'transaction_type': lambda x: x.nunique(),
            'device_used': lambda x: x.nunique(),
            'location': lambda x: x.nunique()
        }).reset_index()
        
        sender_stats.columns = ['sender_account', 'tx_count', 'avg_amount', 'std_amount', 'min_amount', 'max_amount', 'total_spent', 'fraud_count', 'tx_type_variety', 'device_variety', 'location_variety']
        
        # Behavioral scores
        sender_stats['spending_consistency'] = 1 - (sender_stats['std_amount'] / (sender_stats['avg_amount'] + 1))
        sender_stats['activity_score'] = np.log1p(sender_stats['tx_count'])
        sender_stats['fraud_history_ratio'] = sender_stats['fraud_count'] / sender_stats['tx_count']
        
        # Merging back to main dataframe
        self.df = self.df.merge(sender_stats, on='sender_account', how='left')
        print(" Sender behavioral features created")
        
    def create_receiver_features(self):
        print("Creating receiver features...")
        
        receiver_stats = self.df.groupby('receiver_account').agg({
            'transaction_id': 'count',
            'amount': 'mean',
            'is_fraud_int': 'sum'
        }).reset_index()
        
        receiver_stats.columns = ['receiver_account', 'receiver_tx_count', 
                                  'receiver_avg_amount', 'receiver_fraud_count']
        
        receiver_stats['receiver_risk_score'] = (
            receiver_stats['receiver_fraud_count'] / receiver_stats['receiver_tx_count']
        )
        
        self.df = self.df.merge(receiver_stats, on='receiver_account', how='left')
        print(" Receiver features created")
        
    def create_transaction_velocity_features(self):
        print("Creating transaction velocity features...")
        
        # Sort by sender and timestamp
        self.df = self.df.sort_values(['sender_account', 'timestamp'])
        
        # Time difference between consecutive transactions per sender
        self.df['time_since_last_tx'] = self.df.groupby('sender_account')['timestamp'].diff()
        self.df['time_since_last_tx_hours'] = self.df['time_since_last_tx'].dt.total_seconds() / 3600
        self.df['time_since_last_tx_hours'] = self.df['time_since_last_tx_hours'].fillna(999)
        
        # Rapid transaction flag
        self.df['is_rapid_transaction'] = (self.df['time_since_last_tx_hours'] < 0.1).astype(int)
        
        print(" Velocity features created")
        
    def create_amount_based_features(self):
        print("Creating amount-based features...")
        
        # Amount deviation from user's average
        self.df['amount_deviation'] = abs(self.df['amount'] - self.df['avg_amount']) / (self.df['avg_amount'] + 1)
        
        # Large transaction flag
        self.df['is_large_tx'] = (self.df['amount'] > self.df['avg_amount'] * 3).astype(int)
        
        # Round number flag
        self.df['is_round_amount'] = (self.df['amount'] % 100 == 0).astype(int)
        
        print(" Amount features created")
        
    def encode_categorical_features(self):
        print("Encoding categorical features...")
        
        # Transaction type encoding
        tx_type_map = {'deposit': 1, 'withdrawal': 2, 'transfer': 3, 'payment': 4}
        self.df['tx_type_encoded'] = self.df['transaction_type'].map(tx_type_map)
        
        # Device encoding
        device_map = {'mobile': 1, 'web': 2, 'atm': 3, 'pos': 4}
        self.df['device_encoded'] = self.df['device_used'].map(device_map)
        
        # Merchant category encoding
        if 'merchant_category' in self.df.columns:
            self.df['merchant_encoded'] = pd.Categorical(self.df['merchant_category']).codes
        
        print(" Categorical encoding complete")
        
    def handle_missing_values(self):
        print("Handling missing values...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(0)
        
        print(" Missing values handled")
        
    def engineer_all_features(self): #Running complete feature engineering pipeline
        print("\n" + "="*60)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*60 + "\n")
        
        self.parse_timestamps()
        self.create_sender_behavioral_features()
        self.create_receiver_features()
        self.create_transaction_velocity_features()
        self.create_amount_based_features()
        self.encode_categorical_features()
        self.handle_missing_values()
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"Total Features: {len(self.df.columns)}")
        print(f"Original Features: 10")
        print(f"Engineered Features: {len(self.df.columns) - 10}")
        
        return self.df
    
    def get_feature_list(self):

        exclude_cols = ['transaction_id', 'timestamp', 'sender_account', 'receiver_account', 'is_fraud', 'is_fraud_int', 'transaction_type', 'merchant_category', 'location', 'device_used',
                        'time_since_last_tx', 'fraud_type', 'payment_channel', 'ip_address', 'device_hash']
        
        features = [col for col in self.df.columns if col not in exclude_cols]
        return features


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("AFITraining/data/raw/financial_transactions_loaded.csv")
    
    # Initialize feature engineer
    engineer = FeatureEngineer(df)
    
    # Create all features
    df_engineered = engineer.engineer_all_features()
    
    # Get feature list
    feature_cols = engineer.get_feature_list()
    print(f"\nFeatures for modeling ({len(feature_cols)}):")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i}. {feat}")
    
    # Save
    output_path = "AFITraining/data/processed/financial_transactions_engineered.csv"
    df_engineered.to_csv(output_path, index=False)
    print(f"\n Engineered dataset saved to: {output_path}")