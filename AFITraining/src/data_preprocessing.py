# Data Preprocessing and Train Splitting

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    
    def __init__(self, df, feature_cols, sample_size=None):
        
        # Initialize preprocessor
        if sample_size and len(df) > sample_size:
            print(f"\n  Dataset has {len(df):,} rows. Sampling {sample_size:,} rows for memory efficiency...")

            # checking if Stratified sampling to maintain fraud ratio
            fraud_df = df[df['is_fraud'] == True]
            normal_df = df[df['is_fraud'] == False]
            
            # Calculate samples needed
            fraud_ratio = len(fraud_df) / len(df)
            fraud_samples = int(sample_size * fraud_ratio)
            normal_samples = sample_size - fraud_samples
            
            # Sample
            fraud_sampled = fraud_df.sample(n=min(fraud_samples, len(fraud_df)), random_state=42)
            normal_sampled = normal_df.sample(n=normal_samples, random_state=42)
            
            # Combine
            self.df = pd.concat([fraud_sampled, normal_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f" Sampled dataset: {len(self.df):,} rows (Fraud: {self.df['is_fraud'].sum():,})")
        else:
            self.df = df.copy()
            
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        
    def prepare_credit_scoring_data(self, test_size=0.2, random_state=42):
        
        # Prepare data for credit scoring model
        print("\n" + "="*60)
        print("PREPARING CREDIT SCORING DATA")
        print("="*60)
        
        # For credit scoring: fraud as inverse creditworthiness
        X = self.df[self.feature_cols]
        y = self.df['is_fraud'].astype(int)  # Converting boolean to int: 1 = high risk, 0 = low risk
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        print(f"Fraud rate in train: {y_train.mean()*100:.2f}%")
        print(f"Fraud rate in test: {y_test.mean()*100:.2f}%")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_cols)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_cols)
        
        print(" Data scaled using StandardScaler")
        
        return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True)
    
    def prepare_fraud_detection_data(self, test_size=0.2, random_state=42):
    
        # Preparing data for fraud detection model
        print("\n" + "="*60)
        print("PREPARING FRAUD DETECTION DATA")
        print("="*60)
        
        X = self.df[self.feature_cols]
        y = self.df['is_fraud'].astype(int)  # Converting boolean to int
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        print(f"Normal transactions in train: {(y_train==0).sum():,}")
        print(f"Fraudulent transactions in train: {(y_train==1).sum():,}")
        
        # For Isolation Forest: train on normal transactions
        X_train_normal = X_train[y_train == 0]
        print(f" Using {len(X_train_normal):,} normal transactions for training")
        
        # Scale all data
        X_train_normal_scaled = self.scaler.fit_transform(X_train_normal)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to DataFrame
        X_train_normal_scaled = pd.DataFrame(X_train_normal_scaled, columns=self.feature_cols)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_cols)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_cols)
        
        return (X_train_normal_scaled, X_train_scaled, X_test_scaled, 
                y_train.reset_index(drop=True), y_test.reset_index(drop=True))
    
    def save_preprocessor(self, filepath='AFITraining/models/scaler.pkl'):
        #Saving the scaler for later use
        joblib.dump(self.scaler, filepath)
        print(f" Scaler saved to {filepath}")
    
    def get_data_summary(self):
        # Summary
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total samples: {len(self.df):,}")
        print(f"Total features: {len(self.feature_cols)}")
        fraud_rate = self.df['is_fraud'].astype(int).mean()
        print(f"Fraud rate: {fraud_rate*100:.2f}%")
        print(f"\nFeature statistics:")
        print(self.df[self.feature_cols].describe())


if __name__ == "__main__":
    # Loading engineered data
    print("Loading engineered dataset...")
    df = pd.read_csv("AFITraining/data/processed/financial_transactions_engineered.csv")
    print(f" Loaded {len(df):,} rows")
    
    # Defining features
    exclude_cols = ['transaction_id', 'timestamp', 'sender_account', 'receiver_account', 'is_fraud', 'is_fraud_int', 'transaction_type', 
                    'merchant_category', 'location', 'device_used', 'time_since_last_tx', 'fraud_type', 'payment_channel',
                    'ip_address', 'device_hash']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features for modeling")
    
    # To maintains the fraud ratio while reducing memory usage
    preprocessor = DataPreprocessor(df, feature_cols, sample_size=500000)
    
    # Summary
    preprocessor.get_data_summary()
    
    # Prepare credit scoring data
    X_train_cs, X_test_cs, y_train_cs, y_test_cs = preprocessor.prepare_credit_scoring_data()
    
    # Prepare fraud detection data
    X_train_normal_fd, X_train_fd, X_test_fd, y_train_fd, y_test_fd = preprocessor.prepare_fraud_detection_data()
    
    # Save preprocessed data
    import os
    os.makedirs('AFITraining/data/processed', exist_ok=True)
    
    print("\nSaving preprocessed data...")
    
    # Credit scoring data
    X_train_cs.to_csv('AFITraining/data/processed/X_train_credit_scoring.csv', index=False)
    X_test_cs.to_csv('AFITraining/data/processed/X_test_credit_scoring.csv', index=False)
    y_train_cs.to_csv('AFITraining/data/processed/y_train_credit_scoring.csv', index=False, header=['is_fraud'])
    y_test_cs.to_csv('AFITraining/data/processed/y_test_credit_scoring.csv', index=False, header=['is_fraud'])
    
    # Fraud detection data
    X_train_normal_fd.to_csv('AFITraining/data/processed/X_train_normal_fraud.csv', index=False)
    X_train_fd.to_csv('AFITraining/data/processed/X_train_fraud.csv', index=False)
    X_test_fd.to_csv('AFITraining/data/processed/X_test_fraud.csv', index=False)
    y_train_fd.to_csv('AFITraining/data/processed/y_train_fraud.csv', index=False, header=['is_fraud'])
    y_test_fd.to_csv('AFITraining/data/processed/y_test_fraud.csv', index=False, header=['is_fraud'])
    
    # Save scaler
    os.makedirs('AFITraining/models', exist_ok=True)
    preprocessor.save_preprocessor('AFITraining/models/scaler.pkl')
    
    print("\n" + "="*60)
    print(" ALL PREPROCESSED DATA SAVED")
    print("="*60)
    print("Credit Scoring data saved to: AFITraining/data/processed/X_train_credit_scoring.csv")
    print("Fraud Detection data saved to: AFITraining/data/processed/X_train_fraud.csv")
    print("Scaler saved to: AFITraining/models/scaler.pkl")
    print(f"\nFinal dataset size: {len(preprocessor.df):,} rows")
    print(f"Training samples: {len(X_train_cs):,}")
    print(f"Test samples: {len(X_test_cs):,}")