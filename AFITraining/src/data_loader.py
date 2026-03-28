#Data Loading and Initial Exploration

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        # Loading the CSV dataset
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f" Dataset loaded successfully!")
        print(f"  Shape: {self.df.shape}")
        return self.df
    
    def explore_data(self):

        if self.df is None:
            raise ValueError("Data not loaded.") # Call load_data() first.
        
        print("\n" + "="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        
        print(f"\nTotal Transactions: {len(self.df):,}")
        print(f"Features: {self.df.shape[1]}")
        print(f"\nColumn Names and Types:")
        print(self.df.dtypes)
        
        # Missing values
        print("\n" + "-"*60)
        print("MISSING VALUES")
        print("-"*60)
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print(" No missing values found")
        
        # Fraud distribution
        print("\n" + "-"*60)
        print("FRAUD DISTRIBUTION")
        print("-"*60)
        fraud_counts = self.df['is_fraud'].value_counts()
        fraud_pct = self.df['is_fraud'].value_counts(normalize=True) * 100
        print(f"Legitimate: {fraud_counts[False]:,} ({fraud_pct[False]:.2f}%)")
        print(f"Fraudulent: {fraud_counts[True]:,} ({fraud_pct[True]:.2f}%)")
        print(f"Imbalance Ratio: 1:{fraud_counts[False]/fraud_counts[True]:.1f}")
        
        # Transaction types
        print("\n" + "-"*60)
        print("TRANSACTION TYPES")
        print("-"*60)
        print(self.df['transaction_type'].value_counts())
        
        # Amount statistics
        print("\n" + "-"*60)
        print("AMOUNT STATISTICS")
        print("-"*60)
        print(self.df['amount'].describe())
        
        # Device usage
        print("\n" + "-"*60)
        print("DEVICE USAGE")
        print("-"*60)
        print(self.df['device_used'].value_counts())
        
        return self.df
    
    def get_sample_transactions(self, n=5):
       
        print("\n" + "="*60)
        print(f"SAMPLE TRANSACTIONS (First {n})")
        print("="*60)
        print(self.df.head(n).to_string())
        
        print("\n" + "="*60)
        print(f"SAMPLE FRAUDULENT TRANSACTIONS (First {n})")
        print("="*60)
        fraud_samples = self.df[self.df['is_fraud'] == 1].head(n)
        if len(fraud_samples) > 0:
            print(fraud_samples.to_string())
        else:
            print("No fraudulent transactions found in dataset")


if __name__ == "__main__":
    DATA_PATH = "AFITraining/data/raw/financial_transactions.csv"  
    # Initialize loader
    loader = DataLoader(DATA_PATH)
    
    # Load and explore
    df = loader.load_data()
    df = loader.explore_data()
    loader.get_sample_transactions()
    
    print("\n" + "="*60)
    print(" Data loading complete!")
    print("="*60)
    
    # Saving
    print("\nSaving loaded data...")
    df.to_csv("AFITraining/data/raw/financial_transactions_loaded.csv", index=False)