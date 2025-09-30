#!/usr/bin/env python3
"""
Performance vs Explanation Quality Analysis
==========================================

This script analyzes whether model performance increases when explanation quality 
increases, creating detailed visualizations to understand this relationship.
"""

import json
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

# Nature-style utils
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.sar.fig_style import apply_nature_style, save_figure
apply_nature_style()

def load_optimization_data():
    """Load the agentic optimization results"""
    results_path = Path("../results/agentic_optimization_results.json")
    
    if not results_path.exists():
        print("❌ Results file not found.")
        return None
        
    with open(results_path, 'r') as f:
        return json.load(f)

def extract_performance_explanation_data(results):
    """Extract performance and explanation quality data for analysis"""
    data = []
    
    for model_name, model_results in results['model_results'].items():
        if 'all_results' in model_results:
            for i, result in enumerate(model_results['all_results']):
                data.append({
                    'model': model_name,
                    'iteration': result['iteration'],
                    'performance': result['performance'],
                    'explanation_quality': result['explanation_quality'],
                    'combined_score': result['combined_score'],
                    'model_iteration': f"{model_name}_iter_{i}"
                })
    
    return pd.DataFrame(data)

def create_correlation_analysis(df):
    """Create correlation analysis figure"""
    # Nature double-column sized multi-panel
    fig = plt.figure(figsize=(7.2, 6.4))
    
    # Set up the color palette
    colors = {'circular_fingerprint': '#FF6B6B', 'tpot': '#4ECDC4'}
    
    # 1. Overall Scatter Plot with Regression Line
    ax1 = fig.add_subplot(2, 3, 1)
    
    # Plot points for each model
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        ax1.scatter(model_data['explanation_quality'], model_data['performance'], 
                   alpha=0.7, s=20, c=colors.get(model, 'gray'), 
                   label=model.replace('_', ' ').title())
    
    # Add overall regression line
    X = df['explanation_quality'].values.reshape(-1, 1)
    y = df['performance'].values
    reg = LinearRegression().fit(X, y)
    x_range = np.linspace(df['explanation_quality'].min(), df['explanation_quality'].max(), 100)
    y_pred = reg.predict(x_range.reshape(-1, 1))
    ax1.plot(x_range, y_pred, 'r--', alpha=0.8, linewidth=1.0, label=f'Trend (R²={reg.score(X, y):.3f})')
    
    ax1.set_xlabel('Explanation Quality')
    ax1.set_ylabel('Performance')
    ax1.set_title('Performance vs Explanation Quality\nOverall Relationship')
    ax1.legend(frameon=False, fontsize=6)
    ax1.grid(True, alpha=0.2)
    
    # Add correlation coefficient
    corr_coef, p_value = stats.pearsonr(df['explanation_quality'], df['performance'])
    ax1.text(0.05, 0.95, f'r={corr_coef:.3f}\np={p_value:.3f}', 
             transform=ax1.transAxes, fontsize=6,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # 2. Model-by-Model Analysis
    ax2 = fig.add_subplot(2, 3, 2)
    
    model_correlations = {}
    for i, model in enumerate(df['model'].unique()):
        model_data = df[df['model'] == model]
        if len(model_data) > 2:  # Need at least 3 points for correlation
            corr, _ = stats.pearsonr(model_data['explanation_quality'], model_data['performance'])
            model_correlations[model] = corr
            
            # Plot model-specific regression
            X_model = model_data['explanation_quality'].values.reshape(-1, 1)
            y_model = model_data['performance'].values
            reg_model = LinearRegression().fit(X_model, y_model)
            x_model_range = np.linspace(model_data['explanation_quality'].min(), 
                                       model_data['explanation_quality'].max(), 50)
            y_model_pred = reg_model.predict(x_model_range.reshape(-1, 1))
            
            ax2.plot(x_model_range, y_model_pred, color=colors.get(model, 'gray'), 
                    linewidth=1.0, label=f'{model.replace("_", " ").title()}: r={corr:.3f}')
    
    ax2.set_xlabel('Explanation Quality')
    ax2.set_ylabel('Performance')
    ax2.set_title('Model-Specific Correlations')
    ax2.legend(frameon=False, fontsize=6)
    ax2.grid(True, alpha=0.2)
    
    # 3. Iteration Progress Analysis
    ax3 = fig.add_subplot(2, 3, 3)
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('iteration')
        ax3.plot(model_data['iteration'], model_data['explanation_quality'], 
                marker='o', label=f'{model.replace("_", " ").title()} - Quality', 
                color=colors.get(model, 'gray'), alpha=0.7, linewidth=0.9)
        ax3.plot(model_data['iteration'], model_data['performance'], 
                marker='s', label=f'{model.replace("_", " ").title()} - Performance', 
                color=colors.get(model, 'gray'), linestyle='--', alpha=0.7, linewidth=0.9)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Score')
    ax3.set_title('Performance vs Quality Over Time')
    ax3.legend(fontsize=6, frameon=False)
    ax3.grid(True, alpha=0.2)
    
    # 4. Improvement Analysis
    ax4 = fig.add_subplot(2, 3, 4)
    
    improvement_data = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('iteration')
        if len(model_data) > 1:
            perf_improvement = model_data['performance'].iloc[-1] - model_data['performance'].iloc[0]
            qual_improvement = model_data['explanation_quality'].iloc[-1] - model_data['explanation_quality'].iloc[0]
            improvement_data.append({
                'model': model,
                'performance_change': perf_improvement,
                'quality_change': qual_improvement
            })
    
    if improvement_data:
        imp_df = pd.DataFrame(improvement_data)
        
        # Create quadrant plot
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        for _, row in imp_df.iterrows():
            ax4.scatter(row['quality_change'], row['performance_change'], 
                       s=25, c=colors.get(row['model'], 'gray'), alpha=0.7)
            ax4.annotate(row['model'].replace('_', '\n'), 
                        (row['quality_change'], row['performance_change']),
                        xytext=(2, 2), textcoords='offset points', ha='left', fontsize=6)
        
        ax4.set_xlabel('Explanation Quality Change')
        ax4.set_ylabel('Performance Change')
        ax4.set_title('Performance vs Quality Improvement\nQuadrant Analysis')
        ax4.grid(True, alpha=0.2)
        
        # Add quadrant labels
        ax4.text(0.95, 0.95, 'Both Improve', transform=ax4.transAxes, 
                ha='right', va='top', fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        ax4.text(0.05, 0.95, 'Perf↑ Qual↓', transform=ax4.transAxes, 
                ha='left', va='top', fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        ax4.text(0.95, 0.05, 'Perf↓ Qual↑', transform=ax4.transAxes, 
                ha='right', va='bottom', fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        ax4.text(0.05, 0.05, 'Both Decline', transform=ax4.transAxes, 
                ha='left', va='bottom', fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # 5. Distribution Comparison
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Create violin plots
    perf_data = [df[df['model'] == model]['performance'].values for model in df['model'].unique()]
    qual_data = [df[df['model'] == model]['explanation_quality'].values for model in df['model'].unique()]
    
    positions_perf = np.arange(1, len(df['model'].unique()) * 2, 2)
    positions_qual = np.arange(2, len(df['model'].unique()) * 2 + 1, 2)
    
    parts1 = ax5.violinplot(perf_data, positions=positions_perf, widths=0.8)
    parts2 = ax5.violinplot(qual_data, positions=positions_qual, widths=0.8)
    
    # Color the violins
    for pc, color in zip(parts1['bodies'], colors.values()):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    for pc in parts2['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    
    ax5.set_xticks(np.arange(1.5, len(df['model'].unique()) * 2, 2))
    ax5.set_xticklabels([m.replace('_', '\n') for m in df['model'].unique()])
    ax5.set_ylabel('Score')
    ax5.set_title('Score Distributions by Model')
    ax5.grid(True, alpha=0.2)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='gray', alpha=0.7, label='Performance'),
                      Patch(facecolor='lightblue', alpha=0.7, label='Explanation Quality')]
    ax5.legend(handles=legend_elements, frameon=False, fontsize=6)
    
    # 6. Statistical Summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate statistics
    overall_corr, overall_p = stats.pearsonr(df['explanation_quality'], df['performance'])
    
    summary_text = f"""
STATISTICAL SUMMARY

Overall Correlation:
• Pearson r: {overall_corr:.4f}
• p-value: {overall_p:.4f}
• Relationship: {'Positive' if overall_corr > 0 else 'Negative'}

Model-Specific Correlations:
"""
    
    for model, corr in model_correlations.items():
        summary_text += f"• {model.replace('_', ' ').title()}: {corr:.4f}\n"
    
    summary_text += f"""
Key Findings:
• {'Strong' if abs(overall_corr) > 0.7 else 'Moderate' if abs(overall_corr) > 0.3 else 'Weak'} overall correlation
• {'Significant' if overall_p < 0.05 else 'Non-significant'} relationship (α=0.05)
• Best model: {df.loc[df['combined_score'].idxmax(), 'model'].replace('_', ' ').title()}
"""
    
    ax6.text(0.02, 0.98, summary_text, transform=ax6.transAxes, 
            fontsize=6.5, va='top', family='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    fig.tight_layout()
    
    # Save the figure (1200 dpi multi-format)
    output_base = Path("performance_explanation_correlation_analysis.png")
    save_figure(fig, output_base, dpi=1200, formats=("tiff","png","pdf"))
    print(f"Saved correlation analysis to: {output_base.with_suffix('')}.[tiff|png|pdf]")
    
    return model_correlations, overall_corr, overall_p

def create_improvement_tracking_figure(df):
    """Create a detailed figure tracking improvement patterns"""
    fig = plt.figure(figsize=(7.2, 6.4))
    
    colors = {'circular_fingerprint': '#FF6B6B', 'tpot': '#4ECDC4'}
    
    # 1. Sequential Improvement Analysis
    ax1 = fig.add_subplot(2, 3, 1)
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('iteration')
        
        # Calculate rolling improvements
        perf_rolling = model_data['performance'].rolling(window=2, min_periods=1).mean()
        qual_rolling = model_data['explanation_quality'].rolling(window=2, min_periods=1).mean()
        
        ax1.plot(model_data['iteration'], perf_rolling, 
                marker='o', label=f'{model.replace("_", " ").title()} Performance', 
                color=colors.get(model, 'gray'), linewidth=1.0)
        ax1.plot(model_data['iteration'], qual_rolling, 
                marker='s', label=f'{model.replace("_", " ").title()} Quality', 
                color=colors.get(model, 'gray'), linestyle='--', linewidth=1.0, alpha=0.7)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Rolling Average Score')
    ax1.set_title('Rolling Average Performance vs Quality')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6, frameon=False)
    ax1.grid(True, alpha=0.2)
    
    # 2. Best Score Tracking
    ax2 = fig.add_subplot(2, 3, 2)
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('iteration')
        
        # Track cumulative best scores
        best_perf = model_data['performance'].cummax()
        best_qual = model_data['explanation_quality'].cummax()
        
        ax2.plot(model_data['iteration'], best_perf, 
                marker='o', label=f'{model.replace("_", " ").title()} Best Perf', 
                color=colors.get(model, 'gray'), linewidth=1.0)
        ax2.plot(model_data['iteration'], best_qual, 
                marker='s', label=f'{model.replace("_", " ").title()} Best Qual', 
                color=colors.get(model, 'gray'), linestyle='--', linewidth=1.0, alpha=0.7)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cumulative Best Score')
    ax2.set_title('Best Score Evolution')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6, frameon=False)
    ax2.grid(True, alpha=0.2)
    
    # 3. Improvement Momentum
    ax3 = fig.add_subplot(2, 3, 3)
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('iteration')
        if len(model_data) > 1:
            # Calculate differences between consecutive iterations
            perf_diff = model_data['performance'].diff()
            qual_diff = model_data['explanation_quality'].diff()
            
            ax3.scatter(qual_diff.dropna(), perf_diff.dropna(), 
                       s=20, alpha=0.7, c=colors.get(model, 'gray'), 
                       label=model.replace('_', ' ').title())
    
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Quality Change (Iteration to Iteration)')
    ax3.set_ylabel('Performance Change (Iteration to Iteration)')
    ax3.set_title('Improvement Momentum Analysis')
    ax3.legend(frameon=False, fontsize=6)
    ax3.grid(True, alpha=0.2)
    
    # 4. Model Efficiency Analysis
    ax4 = fig.add_subplot(2, 3, 4)
    
    efficiency_data = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        avg_perf = model_data['performance'].mean()
        avg_qual = model_data['explanation_quality'].mean()
        std_combined = model_data['combined_score'].std()
        
        efficiency_data.append({
            'model': model,
            'avg_performance': avg_perf,
            'avg_quality': avg_qual,
            'stability': 1 / (std_combined + 0.001)  # Higher is more stable
        })
    
    eff_df = pd.DataFrame(efficiency_data)
    
    # Create bubble plot
    for _, row in eff_df.iterrows():
        ax4.scatter(row['avg_quality'], row['avg_performance'], 
                   s=row['stability'] * 80, alpha=0.6, 
                   c=colors.get(row['model'], 'gray'))
        ax4.annotate(row['model'].replace('_', '\n'), 
                    (row['avg_quality'], row['avg_performance']),
                    xytext=(2, 2), textcoords='offset points', ha='left', fontsize=6)
    
    ax4.set_xlabel('Average Explanation Quality')
    ax4.set_ylabel('Average Performance')
    ax4.set_title('Model Efficiency Analysis\n(Bubble size = Stability)')
    ax4.grid(True, alpha=0.2)
    
    # 5. Quality-Performance Trade-off Analysis
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Calculate trade-off ratios
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        # Quality to Performance ratio
        qp_ratio = model_data['explanation_quality'] / model_data['performance']
        
        ax5.hist(qp_ratio.dropna(), alpha=0.7, bins=10, 
                label=f'{model.replace("_", " ").title()}', 
                color=colors.get(model, 'gray'))
    
    ax5.axvline(x=1, color='r', linestyle='--', alpha=0.8, label='Equal Quality/Performance')
    ax5.set_xlabel('Quality/Performance Ratio')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Quality-Performance Trade-off Distribution')
    ax5.legend(frameon=False, fontsize=6)
    ax5.grid(True, alpha=0.2)
    
    # 6. Optimization Trajectory
    ax6 = fig.add_subplot(2, 3, 6)
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('iteration')
        
        # Create trajectory plot
        ax6.plot(model_data['explanation_quality'], model_data['performance'], 
                marker='o', markersize=3, linewidth=1.0, alpha=0.7,
                color=colors.get(model, 'gray'),
                label=model.replace('_', ' ').title())
        
        # Mark start and end points
        if len(model_data) > 0:
            ax6.scatter(model_data['explanation_quality'].iloc[0], 
                       model_data['performance'].iloc[0], 
                       s=25, marker='s', color=colors.get(model, 'gray'), 
                       alpha=0.9, edgecolors='black', linewidth=0.6, label='_nolegend_')
            ax6.scatter(model_data['explanation_quality'].iloc[-1], 
                       model_data['performance'].iloc[-1], 
                       s=35, marker='*', color=colors.get(model, 'gray'), 
                       alpha=0.9, edgecolors='black', linewidth=0.6, label='_nolegend_')
    
    ax6.set_xlabel('Explanation Quality')
    ax6.set_ylabel('Performance')
    ax6.set_title('Optimization Trajectory\n(Square=Start, Star=End)')
    ax6.legend(frameon=False, fontsize=6)
    ax6.grid(True, alpha=0.2)
    
    fig.tight_layout()
    
    # Save the figure (1200 dpi multi-format)
    output_base = Path("improvement_tracking_analysis.png")
    save_figure(fig, output_base, dpi=1200, formats=("tiff","png","pdf"))
    print(f"Saved improvement tracking to: {output_base.with_suffix('')}.[tiff|png|pdf]")

def create_summary_report(df, correlations, overall_corr, overall_p):
    """Create a comprehensive summary report"""
    report = []
    report.append("📊 PERFORMANCE vs EXPLANATION QUALITY ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Overall findings
    report.append("🎯 OVERALL FINDINGS")
    report.append("-" * 30)
    report.append(f"Overall correlation coefficient: {overall_corr:.4f}")
    report.append(f"Statistical significance (p-value): {overall_p:.4f}")
    
    if overall_p < 0.05:
        report.append("✅ Statistically significant relationship found!")
    else:
        report.append("⚠️ Relationship not statistically significant")
    
    if overall_corr > 0.3:
        report.append("📈 POSITIVE correlation: Higher explanation quality tends to coincide with higher performance")
    elif overall_corr < -0.3:
        report.append("📉 NEGATIVE correlation: Higher explanation quality tends to coincide with lower performance")
    else:
        report.append("➡️ WEAK correlation: No strong relationship between explanation quality and performance")
    
    report.append("")
    
    # Model-specific analysis
    report.append("🔬 MODEL-SPECIFIC ANALYSIS")
    report.append("-" * 30)
    
    for model, corr in correlations.items():
        model_data = df[df['model'] == model]
        report.append(f"\n{model.upper()}:")
        report.append(f"  • Correlation: {corr:.4f}")
        report.append(f"  • Avg Performance: {model_data['performance'].mean():.4f}")
        report.append(f"  • Avg Quality: {model_data['explanation_quality'].mean():.4f}")
        report.append(f"  • Best Combined Score: {model_data['combined_score'].max():.4f}")
        
        # Improvement analysis
        if len(model_data) > 1:
            perf_improvement = model_data['performance'].max() - model_data['performance'].min()
            qual_improvement = model_data['explanation_quality'].max() - model_data['explanation_quality'].min()
            report.append(f"  • Performance Range: {perf_improvement:.4f}")
            report.append(f"  • Quality Range: {qual_improvement:.4f}")
    
    report.append("")
    
    # Key insights
    report.append("💡 KEY INSIGHTS")
    report.append("-" * 30)
    
    best_model = df.loc[df['combined_score'].idxmax(), 'model']
    best_score = df['combined_score'].max()
    
    report.append(f"🏆 Best performing model: {best_model.upper()}")
    report.append(f"🎯 Best combined score: {best_score:.4f}")
    
    # Check if performance increased with quality
    quality_performance_positive = sum(1 for corr in correlations.values() if corr > 0)
    total_models = len(correlations)
    
    if quality_performance_positive > total_models / 2:
        report.append("✅ CONCLUSION: For most models, higher explanation quality is associated with higher performance")
    else:
        report.append("❌ CONCLUSION: Higher explanation quality does not consistently lead to higher performance")
    
    report.append("")
    
    # Recommendations
    report.append("📋 RECOMMENDATIONS")
    report.append("-" * 30)
    
    if overall_corr > 0.3:
        report.append("• Focus on improving explanation quality as it correlates with better performance")
        report.append("• The agentic optimization successfully balanced both objectives")
    elif overall_corr < -0.3:
        report.append("• Be cautious: improving explanation quality may come at performance cost")
        report.append("• Consider the trade-off carefully based on your specific needs")
    else:
        report.append("• Explanation quality and performance appear to be independent")
        report.append("• Optimize each objective separately based on your priorities")
    
    report.append(f"• {best_model.upper()} shows the best overall balance - consider this architecture")
    
    # Save report
    report_text = "\n".join(report)
    report_path = Path("performance_explanation_analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"📄 Analysis report saved to: {report_path}")
    print("\n" + report_text)
    
    return report_text

def main():
    """Main analysis function"""
    print("📊 Performance vs Explanation Quality Analysis")
    print("=" * 50)
    
    # Load data
    results = load_optimization_data()
    if results is None:
        return
    
    # Extract data
    df = extract_performance_explanation_data(results)
    print(f"📈 Analyzing {len(df)} data points across {df['model'].nunique()} models")
    
    # Create correlation analysis
    correlations, overall_corr, overall_p = create_correlation_analysis(df)
    
    # Create improvement tracking
    create_improvement_tracking_figure(df)
    
    # Create summary report
    create_summary_report(df, correlations, overall_corr, overall_p)
    
    print("\n🎉 Analysis complete! Check the figures/ directory for visualizations.")

if __name__ == "__main__":
    main()
