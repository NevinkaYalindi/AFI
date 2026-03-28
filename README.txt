==============================================================================
AFI - ALTERNATIVE FINANCIAL INTELLIGENCE SYSTEM
Project Documentation & User Guide
==============================================================================

1. OVERVIEW
------------------------------------------------------------------------------
The AFI system utilizes alternative data (transaction history) to compute credit
scores and detect fraud in real-time. This project validates requirements FR03
(Alternative Data Scoring) and NFR3 (Performance/Latency).

2. PREREQUISITES
------------------------------------------------------------------------------
- Python 3.8+
- Required Libraries:
  pandas, numpy, scikit-learn, lightgbm, imbalanced-learn, joblib, 
  streamlit, plotly

To install requirements:
$ pip install pandas numpy scikit-learn lightgbm imbalanced-learn joblib streamlit plotly

3. DIRECTORY STRUCTURE
------------------------------------------------------------------------------
/project_root
  |-- AFITraining/data/                  # Data storage
  |   |-- raw/               # Raw input CSVs
  |   |-- processed/         # Engineered and split datasets
  |-- AFITraining/models/                # Trained models (.pkl, .txt) and metrics (.csv)
  |-- main.py                # Central CLI Entry Point
  |-- README.txt             # This file
  |-- [scripts]              # Individual python modules

4. HOW TO USE (CLI)
------------------------------------------------------------------------------
The system is controlled via `main.py`. Do not run individual scripts unless
debugging specific modules.

Basic Usage:
$ python main.py [COMMAND] [OPTIONS]

5. COMMAND REFERENCE
------------------------------------------------------------------------------

A. Data Processing Pipeline
   Runs data loading, feature engineering, and preprocessing in sequence.
   
   Command:
   $ python main.py pipeline

B. Model Training
   Trains the Machine Learning models.
   
   Train both Credit and Fraud models:
   $ python main.py train
   
   Train only Credit Scoring model:
   $ python main.py train --mode credit
   
   Train only Fraud Detection model:
   $ python main.py train --mode fraud

C. Evaluation
   Generates performance metrics and comparison reports.
   
   Command:
   $ python main.py evaluate

D. Demonstrations
   Run specific scenarios to validate system functionality.
   
   1. API Test (Single Applicant):
      $ python main.py demo --type api
      
   2. Transaction-to-Score Validator (FR03 Demo):
      Shows how specific transaction behaviors affect the score.
      $ python main.py demo --type transaction
      
   3. Integrated System (Full Flow):
      Runs the unified pipeline (Credit Score + Fraud Check).
      $ python main.py demo --type system
      
   4. Batch Processing:
      Processes a batch of applicants to test throughput.
      $ python main.py demo --type batch

E. Dashboard
   Launches the interactive web interface.
   
   Command:
   $ python main.py dashboard

6. TYPICAL WORKFLOW
------------------------------------------------------------------------------
For a fresh run of the system, follow this order:

1. Prepare Data:
   $ python main.py pipeline

2. Train Models:
   $ python main.py train

3. Evaluate Performance:
   $ python main.py evaluate

4. Run Visual Validation:
   $ python main.py demo --type transaction

5. Launch Dashboard:
   $ python main.py dashboard

==============================================================================