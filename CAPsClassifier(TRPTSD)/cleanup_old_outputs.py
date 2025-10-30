#!/usr/bin/env python3
"""
Script to clean up old output files from the outputs directory.
Keeps only the README.md and timestamped run directories.
"""

import os
import glob

def cleanup_old_outputs():
    """Remove old output files, keep only timestamped runs and README"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(script_dir, 'outputs')
    
    if not os.path.exists(outputs_dir):
        print("No outputs directory found.")
        return
    
    # Files to delete (old outputs)
    old_files = [
        'classifier_analysis_subplot1.png',
        'classifier_analysis_subplot2.png',
        'classifier_analysis_subplot3.png',
        'classifier_analysis_subplot4.png',
        'classifier_analysis.png',
        'combined_true_vs_predicted.png',
        'confusion_matrix.png',
        'error_analysis.png',
        'feature_correlation_matrix.png',
        'feature_importance.png',
        'feature_set_comparison.png',
        'patient_specific_feature_importance_comparison.png',
        'patient_specific_results_comparison.png',
        'per_class_metrics.png',
        'RNS_A_B2_patient_errors.png',
        'RNS_A_B2_true_vs_predicted.png',
        'RNS_B_B2_patient_errors.png',
        'RNS_B_B2_true_vs_predicted.png',
        'RNS_D_B1_patient_errors.png',
        'RNS_D_B1_true_vs_predicted.png',
        'RNS_F_B1_patient_errors.png',
        'RNS_F_B1_true_vs_predicted.png',
        'RNS_G_patient_errors.png',
        'RNS_G_true_vs_predicted.png',
    ]
    
    deleted_count = 0
    not_found_count = 0
    
    print("Cleaning up old output files...")
    print("=" * 60)
    
    for filename in old_files:
        filepath = os.path.join(outputs_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"  ✓ Deleted: {filename}")
            deleted_count += 1
        else:
            not_found_count += 1
    
    print("=" * 60)
    print(f"\nCleanup complete!")
    print(f"  Files deleted: {deleted_count}")
    print(f"  Files not found: {not_found_count}")
    print(f"\nKept items:")
    print(f"  - README.md")
    print(f"  - run_* directories (timestamped runs)")

if __name__ == "__main__":
    print("Old Outputs Cleanup Script")
    print("=" * 60)
    print("This will delete old output files from previous model runs.")
    print("Timestamped run directories will be preserved.")
    print("=" * 60)
    
    response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        cleanup_old_outputs()
    else:
        print("Cleanup cancelled.")

