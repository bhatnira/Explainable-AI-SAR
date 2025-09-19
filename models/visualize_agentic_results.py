#!/usr/bin/env python3
"""
Visualize Agentic Parameter Optimization Results
===============================================

This script creates comprehensive visualizations of the agentic optimization
results, showing how the intelligent agent explored parameter spaces and
improved model performance and explanation quality.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd

def load_optimization_results():
    """Load the agentic optimization results"""
    results_path = Path("results/agentic_optimization_results.json")
    
    if not results_path.exists():
        print("❌ Results file not found. Run the agentic optimizer first.")
        return None
        
    with open(results_path, 'r') as f:
        return json.load(f)

def create_comprehensive_visualization(results):
    """Create comprehensive visualization of optimization results"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Extract data for plotting
    models_data = []
    iteration_data = []
    
    for model_name, model_results in results['model_results'].items():
        if 'all_results' in model_results:
            for result in model_results['all_results']:
                models_data.append({
                    'model': model_name,
                    'performance': result['performance'],
                    'explanation_quality': result['explanation_quality'],
                    'combined_score': result['combined_score'],
                    'iteration': result['iteration']
                })
                
                iteration_data.append({
                    'model': model_name,
                    'iteration': result['iteration'],
                    'score': result['combined_score']
                })
    
    df_models = pd.DataFrame(models_data)
    df_iterations = pd.DataFrame(iteration_data)
    
    # 1. Model Performance Comparison
    ax1 = plt.subplot(2, 3, 1)
    model_summary = df_models.groupby('model').agg({
        'performance': 'max',
        'explanation_quality': 'max', 
        'combined_score': 'max'
    })
    
    x = np.arange(len(model_summary.index))
    width = 0.25
    
    bars1 = ax1.bar(x - width, model_summary['performance'], width, 
                   label='Performance', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x, model_summary['explanation_quality'], width,
                   label='Explanation Quality', alpha=0.8, color='lightcoral')
    bars3 = ax1.bar(x + width, model_summary['combined_score'], width,
                   label='Combined Score', alpha=0.8, color='lightgreen')
    
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Score')
    ax1.set_title('🏆 Best Performance by Model Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', '\n') for m in model_summary.index], 
                       rotation=0, ha='center')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # 2. Learning Progress Over Iterations
    ax2 = plt.subplot(2, 3, 2)
    for model in df_iterations['model'].unique():
        model_data = df_iterations[df_iterations['model'] == model]
        ax2.plot(model_data['iteration'], model_data['score'], 
                marker='o', linewidth=2, markersize=6,
                label=model.replace('_', ' ').title())
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Combined Score')
    ax2.set_title('📈 Learning Progress: Score Improvement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance vs Explanation Quality Scatter
    ax3 = plt.subplot(2, 3, 3)
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    for i, model in enumerate(df_models['model'].unique()):
        model_data = df_models[df_models['model'] == model]
        ax3.scatter(model_data['performance'], model_data['explanation_quality'],
                   alpha=0.7, s=100, c=colors[i % len(colors)],
                   label=model.replace('_', ' ').title())
    
    ax3.set_xlabel('Performance Score')
    ax3.set_ylabel('Explanation Quality Score')
    ax3.set_title('🎯 Performance vs Explanation Quality')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add diagonal line for equal performance/quality
    min_val = min(ax3.get_xlim()[0], ax3.get_ylim()[0])
    max_val = max(ax3.get_xlim()[1], ax3.get_ylim()[1])
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # 4. Score Distribution by Model
    ax4 = plt.subplot(2, 3, 4)
    model_names = df_models['model'].unique()
    score_data = [df_models[df_models['model'] == model]['combined_score'].values 
                  for model in model_names]
    
    bp = ax4.boxplot(score_data, labels=[m.replace('_', '\n') for m in model_names],
                    patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Combined Score')
    ax4.set_title('📊 Score Distribution by Model')
    ax4.grid(True, alpha=0.3)
    
    # 5. Best Configuration Summary
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    # Get best overall result
    best_model = results['optimization_summary']['best_overall_model']
    best_score = results['optimization_summary']['best_overall_score']
    best_config = results['model_results'][best_model]['best_configuration']
    
    summary_text = f"""
🏆 OPTIMIZATION SUMMARY

Best Model: {best_model.upper()}
Best Score: {best_score:.3f}

Performance: {best_config['performance']:.3f}
Explanation Quality: {best_config['explanation_quality']:.3f}

🔧 Best Configuration:
"""
    
    for param, value in best_config['parameters'].items():
        summary_text += f"• {param}: {value}\n"
    
    summary_text += f"\n📊 Total Evaluations: {results['optimization_summary']['total_evaluations']}"
    
    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # 6. Agentic Learning State
    ax6 = plt.subplot(2, 3, 6)
    learning_state = results['optimization_summary']['learning_state_final']
    
    # Create a simple visualization of learning state
    metrics = ['Exploration Rate', 'Temperature', 'Total Iterations']
    values = [
        learning_state['exploration_rate'],
        learning_state['temperature'], 
        learning_state['iteration'] / 20  # Normalize for visualization
    ]
    
    bars = ax6.barh(metrics, values, color=['gold', 'orange', 'red'], alpha=0.7)
    ax6.set_xlim(0, 1)
    ax6.set_title('🤖 Final Agentic Learning State')
    
    # Add value labels
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax6.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = Path("results/agentic_optimization_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Comprehensive visualization saved to: {output_path}")
    
    plt.show()
    
    return fig

def create_detailed_report(results):
    """Create a detailed text report of the optimization results"""
    report = []
    report.append("🤖 AGENTIC PARAMETER OPTIMIZATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Summary
    summary = results['optimization_summary']
    report.append(f"🎯 Optimization completed: {summary['timestamp']}")
    report.append(f"📊 Models optimized: {summary['models_optimized']}")
    report.append(f"🔄 Total evaluations: {summary['total_evaluations']}")
    report.append(f"🏆 Best model: {summary['best_overall_model'].upper()}")
    report.append(f"⭐ Best score: {summary['best_overall_score']:.4f}")
    report.append("")
    
    # Model-by-model analysis
    for model_name, model_data in results['model_results'].items():
        if 'best_configuration' not in model_data:
            continue
            
        report.append(f"🔬 {model_name.upper()} ANALYSIS")
        report.append("-" * 40)
        
        best_config = model_data['best_configuration']
        report.append(f"Iterations completed: {model_data['iterations_completed']}")
        report.append(f"Best combined score: {best_config['combined_score']:.4f}")
        report.append(f"  • Performance: {best_config['performance']:.4f}")
        report.append(f"  • Explanation Quality: {best_config['explanation_quality']:.4f}")
        report.append("")
        report.append("Best parameters:")
        for param, value in best_config['parameters'].items():
            report.append(f"  • {param}: {value}")
        report.append("")
        
        # Show improvement over iterations
        if 'all_results' in model_data:
            scores = [r['combined_score'] for r in model_data['all_results']]
            improvement = max(scores) - min(scores)
            report.append(f"Score improvement: {improvement:.4f}")
            report.append(f"Best iteration: {best_config['iteration']}")
        report.append("")
    
    # Learning insights
    learning_state = summary['learning_state_final']
    report.append("🧠 AGENTIC LEARNING INSIGHTS")
    report.append("-" * 40)
    report.append(f"Final exploration rate: {learning_state['exploration_rate']:.4f}")
    report.append(f"Final temperature: {learning_state['temperature']:.4f}")
    report.append(f"Successful patterns found: {len(learning_state['successful_patterns'])}")
    report.append("")
    
    # Key findings
    report.append("💡 KEY FINDINGS")
    report.append("-" * 40)
    
    # Find best performing model
    best_model = summary['best_overall_model']
    best_data = results['model_results'][best_model]['best_configuration']
    
    if best_data['explanation_quality'] > best_data['performance']:
        report.append("✅ Best model achieved higher explanation quality than performance")
    else:
        report.append("⚠️ Best model prioritized performance over explanation quality")
    
    report.append(f"🎯 Optimal balance found: {best_data['combined_score']:.4f}")
    report.append(f"   (60% performance + 40% explanation quality)")
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    report_path = Path("results/agentic_optimization_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"📄 Detailed report saved to: {report_path}")
    print("\n" + report_text)
    
    return report_text

def main():
    """Main function to visualize agentic optimization results"""
    print("🎨 Creating Agentic Optimization Visualizations")
    print("=" * 50)
    
    # Load results
    results = load_optimization_results()
    if results is None:
        return
    
    # Create visualizations
    fig = create_comprehensive_visualization(results)
    
    # Create detailed report
    report = create_detailed_report(results)
    
    print("\n🎉 Visualization and reporting complete!")
    print("Check the results/ directory for outputs.")

if __name__ == "__main__":
    main()
