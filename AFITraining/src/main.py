# Interactive Central Command Center

import subprocess
import sys
import os
import time

def print_header(title):
    print("\n" + "=" * 70)
    print(f" AFI SYSTEM: {title}")
    print("=" * 70)

def run_script(script_name, args=None):

    # Checking if the models are in src or root
    path = script_name if os.path.exists(script_name) else f"AFITraining/src/{script_name}"
    
    if not os.path.exists(path):
        print(f"[!] Error: File '{path}' not found.")
        return False

    if script_name == "dashboard.py":
        cmd = [sys.executable, "-m", "streamlit", "run", path]
    else:
        cmd = [sys.executable, path]
        if args:
            cmd.extend(args)

    print(f"[*] Starting: {script_name}....")
    try:
        subprocess.run(cmd, check=True)
        print(f"[+] Successfully finished: {script_name}")
        return True
    except Exception as e:
        print(f"[!] Error: {e}")
        return False

def interactive_menu():
    while True:
        print_header("MAIN MENU")
        print(" [1] Full Data Pipeline (Load -> Engineer -> Preprocess)")
        print(" [2] Train Models (Train Credit Scoring & Fraud Detection)")
        print(" [3] Run System Evaluation (View Metrics)")
        print(" [4] Run Demos (Transaction Scoring, API, etc.)")
        print(" [5] Launch Interactive Dashboard (Streamlit)")
        print(" [0] Exit")
        
        choice = input("\nSelect an option (0-5): ")

        if choice == '1':
            print_header("RUNNING PIPELINE")
            run_script("data_loader.py")
            run_script("feature_engineering.py")
            run_script("data_preprocessing.py")
            input("\nPipeline Complete. Press Enter to return to menu...")

        elif choice == '2':
            print_header("TRAINING MODELS")
            run_script("hyperparameter_tuning_credit.py")
            run_script("improved_fraud_detection.py")
            input("\nTraining Complete. Press Enter to return to menu...")

        elif choice == '3':
            run_script("evaluation.py")
            input("\nPress Enter to return to menu...")

        elif choice == '4':
            print("\n--- Available Demos ---")
            print(" [1] Transactions to Credit Score") # FR03 Validator
            print(" [2] Real-time Credit API") # Batch Mode
            print(" [3] Integrated System ") # Unified Risk
            demo_choice = input("Select demo: ")
            
            if demo_choice == '1': run_script("transaction_credit_score_demo.py")
            elif demo_choice == '2': run_script("credit_score_api.py", ["--batch", "10"])
            elif demo_choice == '3': run_script("integrated_system.py")
            input("\nPress Enter to return to menu...")

        elif choice == '5':
            print_header("LAUNCHING DASHBOARD")
            run_script("dashboard.py")

        elif choice == '0':
            print("Exiting AFI System.")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    interactive_menu()