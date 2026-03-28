
# Runs complete optimization pipeline and generates comparison report

import subprocess
import sys
import os
from datetime import datetime

print("="*80)
print("AFI - COMPLETE SYSTEM OPTIMIZATION PIPELINE")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Checking required packages are installed
print("\n[STEP 0] Checking dependencies...")
required_packages = ['optuna', 'lightgbm', 'scikit-learn', 'pandas', 'numpy', 'imblearn']

try:
    import optuna
    import lightgbm
    import sklearn
    import pandas
    import numpy
    import imblearn
    print(" All required packages installed")
except ImportError as e:
    print(f" Missing package: {e}")
    print("\nInstall missing packages")
    sys.exit(1)

# Step 1: Credit Scoring Optimization
print("\n" + "="*80)
print("[STEP 1/3] CREDIT SCORING OPTIMIZATION")
print("="*80)
# Goal: Maximize recall to 75%+ while maintaining precision >= 70%
print("\nStarting credit scoring optimization....")

try:
    result = subprocess.run(
        [sys.executable, 'AFITraining/src/hyperparameter_tuning_recall_optimized.py'],
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print("\n Credit scoring optimization completed successfully")
    else:
        print(f"\n Credit scoring optimization failed with code {result.returncode}")
        sys.exit(1)
except Exception as e:
    print(f"\n Error running credit scoring optimization: {e}")
    sys.exit(1)

# Step 2: Fraud Detection Optimization
print("\n" + "="*80)
print("[STEP 2/3] FRAUD DETECTION OPTIMIZATION")
print("="*80)
# Goal: Maximize recall to 70%+ while maintaining precision >= 40%"
print("\nStarting fraud detection optimization...")

try:
    result = subprocess.run(
        [sys.executable, 'AFITraining/src/hyperparameter_tuning_fraud_optimized.py'],
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print("\n Fraud detection optimization completed successfully")
    else:
        print(f"\n Fraud detection optimization failed with code {result.returncode}")
        sys.exit(1)
except Exception as e:
    print(f"\n Error running fraud detection optimization: {e}")
    sys.exit(1)

# Step 3: Generate Comparison Report
print("\n" + "="*80)
print("[STEP 3/3] GENERATING COMPARISON REPORT")
print("="*80)
# Comparing: Original → Improved → Optimized models"
print("\nRunning evaluation...")

try:
    result = subprocess.run(
        [sys.executable, 'AFITraining/src/evaluation.py'],
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print("\n Evaluation completed successfully")
    else:
        print(f"\n  Evaluation completed with warnings (code {result.returncode})")
except Exception as e:
    print(f"\n  Error running evaluation: {e}")

# Summary
print("\n" + "="*80)
print("OPTIMIZATION PIPELINE COMPLETE")
print("="*80)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n RESULTS:")
print("\nGenerated files:")
print("  Credit Scoring:")
print("    - AFITraining/models/optimized_credit_scoring_model.pkl")
print("    - AFITraining/models/optimized_scaler.pkl")
print("    - AFITraining/models/optimized_credit_scoring_metrics.csv")
print("    - AFITraining/models/best_hyperparameters.json")
print("")
print("  Fraud Detection:")
print("    - AFITraining/models/optimized_fraud_lgb.txt")
print("    - AFITraining/models/optimized_fraud_if.pkl")
print("    - AFITraining/models/optimized_fraud_scaler.pkl")
print("    - AFITraining/models/optimized_fraud_detection_metrics.csv")
print("    - AFITraining/models/optimized_ensemble_config.json")
print("")
print("  Evaluation:")
print("    - AFITraining/models/final_evaluation_report.json")
print("\n" + "="*80)
print(" ALL DONE!")
print("="*80)