"""
Generate Excel file with multiple sheets for PPT
Creates a formatted Excel workbook with all tables
"""

import pandas as pd
import numpy as np

# ============================================================================
# DATA
# ============================================================================

# Main Performance Summary
summary_data = {
    'Model': ['DQN', 'Double DQN', 'Dueling DQN', 'PPO', 'A3C', 'DDPG'],
    'Avg_Reward': [24.06, 26.23, 28.43, 122.91, 118.56, 40.57],
    'Std_Dev': [10.77, 4.95, 0.88, 33.25, 35.86, 6.63],
    'CV': [0.448, 0.189, 0.031, 0.271, 0.302, 0.163],
    'Ep_Length': [32.8, 36.5, 40.0, 198.5, 185.2, 95.3],
    'Success_Rate_%': [80, 80, 100, 88, 85, 70],
    'Environment': ['highway-v0', 'highway-v0', 'highway-v0', 'racetrack-v0', 'racetrack-v0', 'racetrack-v0'],
    'Type': ['Value-Based', 'Value-Based', 'Value-Based', 'Policy-Based', 'Policy-Based', 'Policy-Based']
}

# Metric Mapping
mapping_data = {
    'Traditional_ML_Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC/ROC', 'Loss Metric'],
    'RL_Equivalent': [
        'Success Rate / Collision-Free %',
        'Policy Quality (Reward/Action)',
        'Exploration Coverage',
        'Balanced Performance (Reward+Safety)',
        'Cumulative Reward Distribution',
        'TD Error / Bellman Loss'
    ],
    'Description': [
        'Percentage of episodes completed without crashes',
        'Quality of action selection given states',
        'How well agent explores state space',
        'Trade-off between high reward and safety',
        'Spread and consistency of rewards',
        'Prediction error during training'
    ]
}

# Best Performers by Category
rankings_data = {
    'Metric': [
        'Success Rate',
        'Reward Stability',
        'Highest Reward',
        'Episode Length',
        'Consistency (IQR)',
        'Convergence Speed'
    ],
    'Best_Performer': [
        'Dueling DQN',
        'Dueling DQN',
        'PPO',
        'PPO',
        'Dueling DQN',
        'PPO'
    ],
    'Value': [
        '100%',
        'CV=0.031',
        '122.91',
        '198.5 steps',
        'IQR=1.3',
        'Fast'
    ],
    'Category': [
        'Safety',
        'Reliability',
        'Performance',
        'Endurance',
        'Consistency',
        'Efficiency'
    ]
}

# Training Hyperparameters
hyperparams_data = {
    'Model': ['DQN', 'Double DQN', 'Dueling DQN', 'PPO', 'A3C', 'DDPG'],
    'Optimizer': ['Adam', 'Adam', 'Adam', 'Adam', 'RMSprop', 'Adam (x2)'],
    'Learning_Rate': ['1e-4', '1e-4', '1e-4', '5e-4→0', '5e-4→0', '1e-3/2e-3'],
    'Batch_Size': [32, 32, 32, 256, 256, 64],
    'Discount_Gamma': [0.99, 0.99, 0.99, 0.9, 0.99, 0.99],
    'Special_Param': ['τ=0.005', 'τ=0.005', 'τ=0.005', 'ε=0.2', 'Workers', 'τ=0.005'],
    'Episodes': [340, 340, 340, 5000, 5000, 5000],
    'Framework': ['PyTorch', 'PyTorch', 'PyTorch', 'TensorFlow', 'TensorFlow', 'TensorFlow']
}

# ============================================================================
# GENERATE EXCEL
# ============================================================================

def generate_excel_workbook(filename='RL_Performance_Tables.xlsx'):
    """Generate formatted Excel workbook with all tables"""
    
    print(f"Generating Excel workbook: {filename}")
    
    # Create Excel writer
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # Sheet 1: Main Performance Summary
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Performance Summary', index=False)
        print("  ✓ Sheet 1: Performance Summary")
        
        # Sheet 2: Metric Mapping
        df_mapping = pd.DataFrame(mapping_data)
        df_mapping.to_excel(writer, sheet_name='ML vs RL Metrics', index=False)
        print("  ✓ Sheet 2: ML vs RL Metrics")
        
        # Sheet 3: Rankings
        df_rankings = pd.DataFrame(rankings_data)
        df_rankings.to_excel(writer, sheet_name='Rankings', index=False)
        print("  ✓ Sheet 3: Rankings by Category")
        
        # Sheet 4: Hyperparameters
        df_hyperparams = pd.DataFrame(hyperparams_data)
        df_hyperparams.to_excel(writer, sheet_name='Hyperparameters', index=False)
        print("  ✓ Sheet 4: Training Hyperparameters")
        
        # Sheet 5: DQN Only
        df_dqn = df_summary[df_summary['Type'] == 'Value-Based']
        df_dqn.to_excel(writer, sheet_name='DQN Family', index=False)
        print("  ✓ Sheet 5: DQN Family Results")
        
        # Sheet 6: Policy-Based Only
        df_policy = df_summary[df_summary['Type'] == 'Policy-Based']
        df_policy.to_excel(writer, sheet_name='Policy-Based', index=False)
        print("  ✓ Sheet 6: Policy-Based Results")
        
        # Sheet 7: Quick Reference
        quick_ref = pd.DataFrame({
            'Key Insight': [
                'Best Value-Based Model',
                'Best Policy-Based Model',
                'Most Stable Model',
                'Highest Reward',
                'Best Safety Record',
                'Longest Survival',
                'Framework for Value-Based',
                'Framework for Policy-Based'
            ],
            'Answer': [
                'Dueling DQN',
                'PPO',
                'Dueling DQN (CV=0.031)',
                'PPO (122.91)',
                'Dueling DQN (100% success)',
                'PPO (198.5 steps)',
                'PyTorch',
                'TensorFlow/Keras'
            ]
        })
        quick_ref.to_excel(writer, sheet_name='Quick Reference', index=False)
        print("  ✓ Sheet 7: Quick Reference Guide")
    
    print(f"\n✓ Excel workbook created: {filename}")
    print(f"\nYou can now:")
    print(f"  1. Open {filename} in Excel")
    print(f"  2. Copy any table and paste into PowerPoint")
    print(f"  3. Use 'Performance Summary' sheet for main results")
    print(f"  4. All tables are properly formatted and ready to use!")

# ============================================================================
# GENERATE COMPARISON CHART DATA
# ============================================================================

def generate_chart_data(filename='chart_data.csv'):
    """Generate data specifically formatted for charts in PPT"""
    
    # Pivot data for easy charting
    chart_data = pd.DataFrame({
        'Model': ['DQN', 'Double DQN', 'Dueling DQN', 'PPO', 'A3C', 'DDPG'],
        'Reward': [24.06, 26.23, 28.43, 122.91, 118.56, 40.57],
        'StdDev': [10.77, 4.95, 0.88, 33.25, 35.86, 6.63],
        'Length': [32.8, 36.5, 40.0, 198.5, 185.2, 95.3],
        'Success': [80, 80, 100, 88, 85, 70]
    })
    
    chart_data.to_csv(filename, index=False)
    print(f"\n✓ Chart data saved to: {filename}")
    print(f"  Use this to create bar charts in PPT (Insert → Chart → Import Data)")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("EXCEL TABLE GENERATOR FOR PPT")
    print("="*70 + "\n")
    
    # Generate Excel workbook
    generate_excel_workbook()
    
    # Generate chart data
    generate_chart_data()
    
    print("\n" + "="*70)
    print("✓ ALL FILES GENERATED!")
    print("="*70)
    
    # Display summary
    df = pd.DataFrame(summary_data)
    print("\n📊 PERFORMANCE SUMMARY:")
    print(df[['Model', 'Avg_Reward', 'Std_Dev', 'Success_Rate_%']].to_string(index=False))
    
    print("\n🎯 KEY FINDINGS:")
    print("  • Best Safety: Dueling DQN (100% success)")
    print("  • Highest Reward: PPO (122.91)")
    print("  • Most Stable: Dueling DQN (CV=0.031)")
    print("  • Longest Survival: PPO (198.5 steps)")

if __name__ == "__main__":
    main()
