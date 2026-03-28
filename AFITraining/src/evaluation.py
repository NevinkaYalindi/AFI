# Model Evaluation - INCLUDING OPTIMIZED MODELS

import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class FinalModelEvaluation:
    
    def __init__(self):
        self.metrics = {}
        
    def safe_get_metric(self, metrics_dict, *possible_keys):
        # Trying multiple possible column names
        for key in possible_keys:
            if key in metrics_dict:
                return metrics_dict[key]
        return 0.0
    
    def load_all_metrics(self):
        # Loading all available metrics from CSV files
        print("="*70)
        print("LOADING AVAILABLE MODEL METRICS")
        print("="*70)
        
        metrics_files = {
            'original_credit': 'AFITraining/models/credit_scoring_metrics.csv',
            'improved_credit': 'AFITraining/models/improved_credit_scoring_metrics.csv',
            'optimized_credit': 'AFITraining/models/optimized_credit_scoring_metrics.csv',
            'original_fraud': 'AFITraining/models/fraud_detection_metrics.csv',
            'final_fraud': 'AFITraining/models/final_fraud_metrics.csv',
            'optimized_fraud': 'AFITraining/models/optimized_fraud_detection_metrics.csv'
        }
        
        for key, filepath in metrics_files.items():
            try:
                df = pd.read_csv(filepath)
                self.metrics[key] = df.iloc[0].to_dict()
                print(f" Loaded {key:20s}: {filepath}")
            except FileNotFoundError:
                print(f" Not found {key:20s}: {filepath}")
                self.metrics[key] = None
            except Exception as e:
                print(f" Error loading {key:20s}: {e}")
                self.metrics[key] = None
        
        print("\n" + "="*70)
    
    def display_credit_scoring_evolution(self):
        # credit scoring model evolution
        print("\n" + "="*70)
        print("CREDIT SCORING MODEL EVOLUTION")
        print("="*70)
        
        # Checking what's available
        models_to_show = []
        
        if self.metrics.get('original_credit'):
            models_to_show.append(('original_credit', 'Original'))
        
        if self.metrics.get('improved_credit'):
            models_to_show.append(('improved_credit', 'Improved (SMOTE)'))
        
        if self.metrics.get('optimized_credit'):
            models_to_show.append(('optimized_credit', 'Optimized (Optuna)'))
        
        if not models_to_show:
            print("\n No credit scoring metrics found")
            print("\nExpected files:")
            print("  - AFITraining/models/credit_scoring_metrics.csv (original)")
            print("  - AFITraining/models/improved_credit_scoring_metrics.csv (improved)")
            print("  - AFITraining/models/optimized_credit_scoring_metrics.csv (optimized)")
            return
        
        # Display table
        print("\n" + "-"*80)
        print(f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'AUC-ROC':>10}")
        print("-"*80)
        
        for model_key, model_label in models_to_show:
            m = self.metrics[model_key]
            
            # getting metrics with multiple possible column names
            accuracy = self.safe_get_metric(m, 'accuracy', 'Accuracy')
            precision = self.safe_get_metric(m, 'precision', 'Precision')
            recall = self.safe_get_metric(m, 'recall', 'Recall')
            f1 = self.safe_get_metric(m, 'f1_score', 'f1', 'F1-Score', 'f1score')
            auc = self.safe_get_metric(m, 'auc_roc', 'auc', 'AUC-ROC', 'auc_score')
            
            print(f"{model_label:<30} {accuracy*100:>9.2f}% {precision*100:>9.2f}% "
                  f"{recall*100:>9.2f}% {f1:>10.4f} {auc:>10.4f}")
        
        # Show improvement progression
        if len(models_to_show) > 1:
            print("\n" + "-"*80)
            print("IMPROVEMENT PROGRESSION")
            print("-"*80)
            
            for i in range(1, len(models_to_show)):
                prev_key, prev_label = models_to_show[i-1]
                curr_key, curr_label = models_to_show[i]
                
                prev_m = self.metrics[prev_key]
                curr_m = self.metrics[curr_key]
                
                prev_recall = self.safe_get_metric(prev_m, 'recall', 'Recall')
                curr_recall = self.safe_get_metric(curr_m, 'recall', 'Recall')
                prev_precision = self.safe_get_metric(prev_m, 'precision', 'Precision')
                curr_precision = self.safe_get_metric(curr_m, 'precision', 'Precision')
                
                recall_change = (curr_recall - prev_recall) * 100
                precision_change = (curr_precision - prev_precision) * 100
                
                print(f"\n{prev_label} -> {curr_label}:")
                print(f"  Recall:    {prev_recall*100:.2f}% -> {curr_recall*100:.2f}% ({recall_change:+.2f}%)")
                print(f"  Precision: {prev_precision*100:.2f}% -> {curr_precision*100:.2f}% ({precision_change:+.2f}%)")
        
        print("\n" + "-"*80)
        print("TARGET ACHIEVEMENT - CREDIT SCORING")
        print("-"*80)
        
        # Using latest available model
        latest_key, latest_label = models_to_show[-1]
        m = self.metrics[latest_key]
        
        precision = self.safe_get_metric(m, 'precision', 'Precision')
        recall = self.safe_get_metric(m, 'recall', 'Recall')
        
        print(f"\nCurrent Model: {latest_label}")
        print(f"\nRecall Target: 75%")
        if recall >= 0.75:
            print(f"   ACHIEVED: {recall*100:.2f}% (+{(recall-0.75)*100:.2f}% above target)")
        else:
            print(f"   MISSED: {recall*100:.2f}% ({(recall-0.75)*100:.2f}% below target)")
        
        print(f"\nPrecision Target: 70%")
        if precision >= 0.70:
            print(f"   ACHIEVED: {precision*100:.2f}% (+{(precision-0.70)*100:.2f}% above target)")
        else:
            print(f"    BELOW TARGET: {precision*100:.2f}% ({(precision-0.70)*100:.2f}% below target)")
        
        # Overall improvement
        if len(models_to_show) > 1 and self.metrics.get('original_credit'):
            orig = self.metrics['original_credit']
            orig_precision = self.safe_get_metric(orig, 'precision', 'Precision')
            orig_recall = self.safe_get_metric(orig, 'recall', 'Recall')
            
            recall_gain = (recall - orig_recall) * 100
            precision_change = (precision - orig_precision) * 100
            
            print(f"\n Total Improvement from Original:")
            print(f"   Recall:    {orig_recall*100:.2f}% -> {recall*100:.2f}% ({recall_gain:+.2f}%)")
            print(f"   Precision: {orig_precision*100:.2f}% -> {precision*100:.2f}% ({precision_change:+.2f}%)")
    
    def display_fraud_detection_evolution(self):
        # fraud detection model evolution
        print("\n" + "="*70)
        print("FRAUD DETECTION MODEL EVOLUTION")
        print("="*70)
        
        # Checking what's available
        models_to_show = []
        
        if self.metrics.get('original_fraud'):
            models_to_show.append(('original_fraud', 'Original (IF)'))
        
        if self.metrics.get('final_fraud'):
            models_to_show.append(('final_fraud', 'Improved (LGB+SMOTE)'))
        
        if self.metrics.get('optimized_fraud'):
            models_to_show.append(('optimized_fraud', 'Optimized (Optuna Ensemble)'))
        
        if not models_to_show:
            print("\n No fraud detection metrics found")
            print("\nExpected files:")
            print("  - AFITraining/models/fraud_detection_metrics.csv (original)")
            print("  - AFITraining/models/final_fraud_metrics.csv (improved)")
            print("  - AFITraining/models/optimized_fraud_detection_metrics.csv (optimized)")
            return
        
        # Display table
        print("\n" + "-"*80)
        print(f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'AUC-ROC':>10}")
        print("-"*80)
        
        for model_key, model_label in models_to_show:
            m = self.metrics[model_key]
            
            # getting metrics
            accuracy = self.safe_get_metric(m, 'accuracy', 'Accuracy')
            precision = self.safe_get_metric(m, 'precision', 'Precision')
            recall = self.safe_get_metric(m, 'recall', 'Recall')
            f1 = self.safe_get_metric(m, 'f1_score', 'f1', 'F1-Score', 'f1score')
            auc = self.safe_get_metric(m, 'auc_roc', 'auc', 'AUC-ROC', 'auc_score')
            
            print(f"{model_label:<30} {accuracy*100:>9.2f}% {precision*100:>9.2f}% "
                  f"{recall*100:>9.2f}% {f1:>10.4f} {auc:>10.4f}")
        
        # Show improvement progression
        if len(models_to_show) > 1:
            print("\n" + "-"*80)
            print("IMPROVEMENT PROGRESSION")
            print("-"*80)
            
            for i in range(1, len(models_to_show)):
                prev_key, prev_label = models_to_show[i-1]
                curr_key, curr_label = models_to_show[i]
                
                prev_m = self.metrics[prev_key]
                curr_m = self.metrics[curr_key]
                
                prev_recall = self.safe_get_metric(prev_m, 'recall', 'Recall')
                curr_recall = self.safe_get_metric(curr_m, 'recall', 'Recall')
                prev_precision = self.safe_get_metric(prev_m, 'precision', 'Precision')
                curr_precision = self.safe_get_metric(curr_m, 'precision', 'Precision')
                
                recall_change = (curr_recall - prev_recall) * 100
                precision_change = (curr_precision - prev_precision) * 100
                
                print(f"\n{prev_label} -> {curr_label}:")
                print(f"  Recall:    {prev_recall*100:.2f}% -> {curr_recall*100:.2f}% ({recall_change:+.2f}%)")
                print(f"  Precision: {prev_precision*100:.2f}% -> {curr_precision*100:.2f}% ({precision_change:+.2f}%)")
     
        print("\n" + "-"*80)
        print("TARGET ACHIEVEMENT - FRAUD DETECTION")
        print("-"*80)
        
        latest_key, latest_label = models_to_show[-1]
        m = self.metrics[latest_key]
        
        precision = self.safe_get_metric(m, 'precision', 'Precision')
        recall = self.safe_get_metric(m, 'recall', 'Recall')
        fpr = self.safe_get_metric(m, 'false_positive_rate', 'fpr', 'FPR')
        
        print(f"\nCurrent Model: {latest_label}")
        print(f"\nRecall Target: 70%")
        if recall >= 0.70:
            print(f"   ACHIEVED: {recall*100:.2f}% (+{(recall-0.70)*100:.2f}% above target)")
        else:
            print(f"   MISSED: {recall*100:.2f}% ({(recall-0.70)*100:.2f}% below target)")
        
        print(f"\nPrecision Target: 40%")
        if precision >= 0.40:
            print(f"   ACHIEVED: {precision*100:.2f}% (+{(precision-0.40)*100:.2f}% above target)")
        else:
            print(f"   MISSED: {precision*100:.2f}% ({(precision-0.40)*100:.2f}% below target)")
        
        if fpr > 0:
            print(f"\n False Positive Rate: {fpr*100:.2f}%")
        
        # Overall improvement
        if len(models_to_show) > 1 and self.metrics.get('original_fraud'):
            orig = self.metrics['original_fraud']
            orig_precision = self.safe_get_metric(orig, 'precision', 'Precision')
            orig_recall = self.safe_get_metric(orig, 'recall', 'Recall')
            
            recall_gain = (recall - orig_recall) * 100
            precision_change = (precision - orig_precision) * 100
            
            print(f"\n Total Improvement from Original:")
            print(f"   Recall:    {orig_recall*100:.2f}% -> {recall*100:.2f}% ({recall_gain:+.2f}%)")
            print(f"   Precision: {orig_precision*100:.2f}% -> {precision*100:.2f}% ({precision_change:+.2f}%)")
    
    def display_overall_summary(self):
        print("\n" + "="*70)
        print("OVERALL SYSTEM STATUS")
        print("="*70)
        
        # Credit Scoring Status
        print("\n CREDIT SCORING:")
        
        # Use optimized if available, else improved
        credit_model = None
        credit_label = None
        
        if self.metrics.get('optimized_credit'):
            credit_model = self.metrics['optimized_credit']
            credit_label = "Optimized (Optuna)"
        elif self.metrics.get('improved_credit'):
            credit_model = self.metrics['improved_credit']
            credit_label = "Improved (SMOTE)"
        
        if credit_model:
            precision = self.safe_get_metric(credit_model, 'precision', 'Precision')
            recall = self.safe_get_metric(credit_model, 'recall', 'Recall')
            auc = self.safe_get_metric(credit_model, 'auc_roc', 'auc', 'AUC-ROC')
            
            print(f"  Model: {credit_label}")
            print(f"  Recall:    {recall*100:.2f}% {'' if recall >= 0.75 else ''} (Target: 75%)")
            print(f"  Precision: {precision*100:.2f}% {'' if precision >= 0.70 else ' '} (Target: 70%)")
            print(f"  AUC-ROC:   {auc:.4f}")
        else:
            print("   No metrics available")
        
        # Fraud Detection Status
        print("\n FRAUD DETECTION:")
        
        # Use optimized if available, else final
        fraud_model = None
        fraud_label = None
        
        if self.metrics.get('optimized_fraud'):
            fraud_model = self.metrics['optimized_fraud']
            fraud_label = "Optimized (Optuna Ensemble)"
        elif self.metrics.get('final_fraud'):
            fraud_model = self.metrics['final_fraud']
            fraud_label = "Improved (LGB+SMOTE)"
        
        if fraud_model:
            precision = self.safe_get_metric(fraud_model, 'precision', 'Precision')
            recall = self.safe_get_metric(fraud_model, 'recall', 'Recall')
            auc = self.safe_get_metric(fraud_model, 'auc_roc', 'auc', 'AUC-ROC')
            fpr = self.safe_get_metric(fraud_model, 'false_positive_rate', 'fpr', 'FPR')
            
            print(f"  Model: {fraud_label}")
            print(f"  Recall:    {recall*100:.2f}% {'' if recall >= 0.70 else ''} (Target: 70%)")
            print(f"  Precision: {precision*100:.2f}% {'' if precision >= 0.40 else ''} (Target: 40%)")
            print(f"  AUC-ROC:   {auc:.4f}")
            if fpr > 0:
                print(f"  FPR:       {fpr*100:.2f}%")
        else:
            print("  ✗ No metrics available")
        
        # Overall Status
        print("\n" + "="*70)
        print(" OVERALL TARGET ACHIEVEMENT")
        print("="*70)
        
        credit_recall_ok = False
        credit_precision_ok = False
        fraud_recall_ok = False
        fraud_precision_ok = False
        
        if credit_model:
            recall = self.safe_get_metric(credit_model, 'recall', 'Recall')
            precision = self.safe_get_metric(credit_model, 'precision', 'Precision')
            credit_recall_ok = (recall >= 0.75)
            credit_precision_ok = (precision >= 0.70)
        
        if fraud_model:
            recall = self.safe_get_metric(fraud_model, 'recall', 'Recall')
            precision = self.safe_get_metric(fraud_model, 'precision', 'Precision')
            fraud_recall_ok = (recall >= 0.70)
            fraud_precision_ok = (precision >= 0.40)
        
        print("\nCredit Scoring:")
        print(f"  Recall ≥75%:     {' MET' if credit_recall_ok else ' NOT MET'}")
        print(f"  Precision ≥70%:  {' MET' if credit_precision_ok else '  NOT MET'}")
        
        print("\nFraud Detection:")
        print(f"  Recall ≥70%:     {' MET' if fraud_recall_ok else ' NOT MET'}")
        print(f"  Precision ≥40%:  {' MET' if fraud_precision_ok else ' NOT MET'}")
        
        # Summary
        total_targets = 4
        met_targets = sum([credit_recall_ok, credit_precision_ok, fraud_recall_ok, fraud_precision_ok])
        
        print(f"\n{'='*70}")
        print(f"SUMMARY: {met_targets}/{total_targets} targets met ({met_targets/total_targets*100:.0f}%)")
        print(f"{'='*70}")
        
        if met_targets == 4:
            print("\n EXCELLENT! All targets achieved!")
        elif met_targets >= 3:
            print("\n GOOD! Most targets achieved!")
        elif met_targets >= 2:
            print("\n  PARTIAL SUCCESS - Some targets met")
        else:
            print("\n NEEDS IMPROVEMENT - Most targets not met")
        
    
    def save_final_report(self):
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'system_name': 'AFI - Alternative Financial Intelligence',
            'available_metrics': [k for k, v in self.metrics.items() if v is not None],
            'credit_scoring': {},
            'fraud_detection': {},
            'summary': {}
        }
        
        # Credit scoring
        credit_model = self.metrics.get('optimized_credit') or self.metrics.get('improved_credit')
        credit_version = 'Optimized (Optuna)' if self.metrics.get('optimized_credit') else 'Improved (SMOTE)'
        
        if credit_model:
            recall = self.safe_get_metric(credit_model, 'recall', 'Recall')
            precision = self.safe_get_metric(credit_model, 'precision', 'Precision')
            
            report['credit_scoring'] = {
                'model_version': credit_version,
                'accuracy': float(self.safe_get_metric(credit_model, 'accuracy', 'Accuracy')),
                'recall': float(recall),
                'precision': float(precision),
                'f1_score': float(self.safe_get_metric(credit_model, 'f1_score', 'f1', 'F1-Score')),
                'auc_roc': float(self.safe_get_metric(credit_model, 'auc_roc', 'auc', 'AUC-ROC')),
                'recall_target': 0.75,
                'precision_target': 0.70,
                'targets_met': {
                    'recall': bool(recall >= 0.75),
                    'precision': bool(precision >= 0.70)
                }
            }
        
        # Fraud detection
        fraud_model = self.metrics.get('optimized_fraud') or self.metrics.get('final_fraud')
        fraud_version = 'Optimized (Optuna Ensemble)' if self.metrics.get('optimized_fraud') else 'Improved (LGB+SMOTE)'
        
        if fraud_model:
            recall = self.safe_get_metric(fraud_model, 'recall', 'Recall')
            precision = self.safe_get_metric(fraud_model, 'precision', 'Precision')
            
            report['fraud_detection'] = {
                'model_version': fraud_version,
                'accuracy': float(self.safe_get_metric(fraud_model, 'accuracy', 'Accuracy')),
                'recall': float(recall),
                'precision': float(precision),
                'f1_score': float(self.safe_get_metric(fraud_model, 'f1_score', 'f1', 'F1-Score')),
                'auc_roc': float(self.safe_get_metric(fraud_model, 'auc_roc', 'auc', 'AUC-ROC')),
                'false_positive_rate': float(self.safe_get_metric(fraud_model, 'false_positive_rate', 'fpr', 'FPR')),
                'recall_target': 0.70,
                'precision_target': 0.40,
                'targets_met': {
                    'recall': bool(recall >= 0.70),
                    'precision': bool(precision >= 0.40)
                }
            }
        
        # Summary
        targets_met = 0
        if report.get('credit_scoring'):
            targets_met += sum(report['credit_scoring']['targets_met'].values())
        if report.get('fraud_detection'):
            targets_met += sum(report['fraud_detection']['targets_met'].values())
        
        report['summary'] = {
            'total_targets': 4,
            'targets_met': targets_met,
            'success_rate': f"{targets_met/4*100:.1f}%",
            'status': 'EXCELLENT' if targets_met == 4 else 'GOOD' if targets_met >= 3 else 'PARTIAL'
        }
        
        os.makedirs('AFITraining/models', exist_ok=True)
        with open('AFITraining/models/final_evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n Final report saved: AFITraining/models/final_evaluation_report.json")
        return report


if __name__ == "__main__":
    print("="*70)
    print("AFI - COMPREHENSIVE MODEL EVALUATION")
    print("Comparing: Original -> Improved -> Optimized Models")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    evaluator = FinalModelEvaluation()
    evaluator.load_all_metrics()
    evaluator.display_credit_scoring_evolution()
    evaluator.display_fraud_detection_evolution()
    evaluator.display_overall_summary()
    evaluator.save_final_report()
    
    print("\n" + "="*70)
    print(" EVALUATION COMPLETE")
    print("="*70)