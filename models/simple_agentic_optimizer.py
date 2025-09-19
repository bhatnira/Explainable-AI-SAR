"""
Simple Agentic Parameter Optimizer - Demo Version
================================================

A streamlined agentic approach to optimize model parameters for better
explanation quality. This version focuses on practical parameter tuning
with immediate results.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import itertools

class SimpleAgenticOptimizer:
    """
    Simplified agentic optimizer that intelligently explores parameter spaces
    to improve explanation quality metrics.
    """
    
    def __init__(self):
        self.optimization_history = []
        self.best_configurations = {}
        self.learning_state = {
            'exploration_rate': 0.7,
            'temperature': 1.0,
            'successful_patterns': {},
            'iteration': 0
        }
        
        print("🤖 Simple Agentic Parameter Optimizer Initialized")
    
    def define_parameter_spaces(self):
        """Define parameter search spaces for each model type."""
        return {
            'circular_fingerprint': {
                'radius': [1, 2, 3, 4],
                'nBits': [512, 1024, 2048, 4096],
                'useFeatures': [True, False],
                'useChirality': [True, False]
            },
            'graphconv': {
                'graph_conv_layers': [
                    [32, 32], [64, 64], [128, 128],
                    [32, 64], [64, 128], [128, 64],
                    [64, 64, 64], [32, 64, 32]
                ],
                'dense_layer_size': [64, 128, 256],
                'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                'learning_rate': [0.0005, 0.001, 0.005]
            },
            'chemberta': {
                'max_length': [128, 256, 512],
                'learning_rate': [1e-5, 2e-5, 5e-5],
                'batch_size': [8, 16, 32],
                'num_epochs': [3, 5, 10]
            }
        }
    
    def intelligent_parameter_selection(self, model_type: str, num_candidates: int = 8) -> List[Dict]:
        """
        Intelligently select parameter combinations based on learning history.
        """
        parameter_space = self.define_parameter_spaces()[model_type]
        candidates = []
        
        # Strategy 1: Best known configurations (exploitation)
        if model_type in self.best_configurations:
            best_config = self.best_configurations[model_type]
            candidates.append(best_config['parameters'])
            
            # Generate variations around best configuration
            for _ in range(2):
                variation = self._create_parameter_variation(best_config['parameters'], parameter_space)
                candidates.append(variation)
        
        # Strategy 2: Successful pattern mining
        successful_configs = [h for h in self.optimization_history 
                            if h['model_type'] == model_type and h['combined_score'] > 0.6]
        
        if successful_configs:
            # Analyze successful patterns
            patterns = self._extract_successful_patterns(successful_configs, model_type)
            
            # Generate new configurations based on patterns
            for pattern in patterns[:2]:
                new_config = self._generate_from_pattern(pattern, parameter_space)
                candidates.append(new_config)
        
        # Strategy 3: Smart random exploration
        remaining_slots = num_candidates - len(candidates)
        for _ in range(remaining_slots):
            if np.random.random() < self.learning_state['exploration_rate']:
                # Pure exploration
                random_config = self._generate_random_config(parameter_space)
            else:
                # Guided exploration based on learning
                random_config = self._generate_guided_config(parameter_space, model_type)
            
            candidates.append(random_config)
        
        return candidates[:num_candidates]
    
    def _create_parameter_variation(self, base_config: Dict, parameter_space: Dict) -> Dict:
        """Create a variation of a base configuration."""
        variation = base_config.copy()
        
        # Randomly modify 1-2 parameters
        params_to_modify = np.random.choice(
            list(base_config.keys()), 
            size=min(2, len(base_config)), 
            replace=False
        )
        
        for param in params_to_modify:
            if param in parameter_space:
                # Choose a different value from the parameter space
                current_value = base_config[param]
                available_values = [v for v in parameter_space[param] if v != current_value]
                if available_values:
                    variation[param] = np.random.choice(available_values)
        
        return variation
    
    def _extract_successful_patterns(self, successful_configs: List[Dict], model_type: str) -> List[Dict]:
        """Extract patterns from successful configurations."""
        patterns = []
        
        if not successful_configs:
            return patterns
        
        # Find common parameter values in successful configurations
        parameter_counts = {}
        
        for config in successful_configs:
            for param, value in config['parameters'].items():
                if param not in parameter_counts:
                    parameter_counts[param] = {}
                if str(value) not in parameter_counts[param]:
                    parameter_counts[param][str(value)] = 0
                parameter_counts[param][str(value)] += 1
        
        # Create patterns from frequently successful parameter values
        for param, value_counts in parameter_counts.items():
            total_configs = len(successful_configs)
            for value_str, count in value_counts.items():
                if count / total_configs > 0.5:  # Appears in > 50% of successful configs
                    patterns.append({param: eval(value_str) if value_str.startswith('[') else value_str})
        
        return patterns
    
    def _generate_from_pattern(self, pattern: Dict, parameter_space: Dict) -> Dict:
        """Generate a configuration based on a successful pattern."""
        config = {}
        
        # Start with the pattern
        for param, value in pattern.items():
            if param in parameter_space:
                config[param] = value
        
        # Fill in missing parameters
        for param, values in parameter_space.items():
            if param not in config:
                config[param] = np.random.choice(values)
        
        return config
    
    def _generate_random_config(self, parameter_space: Dict) -> Dict:
        """Generate a completely random configuration."""
        config = {}
        for param, values in parameter_space.items():
            config[param] = np.random.choice(values)
        return config
    
    def _generate_guided_config(self, parameter_space: Dict, model_type: str) -> Dict:
        """Generate a configuration guided by domain knowledge."""
        config = {}
        
        # Apply domain-specific heuristics
        if model_type == 'circular_fingerprint':
            # For fingerprints, moderate radius and size usually work well
            config['radius'] = np.random.choice([2, 3])
            config['nBits'] = np.random.choice([1024, 2048])
            config['useFeatures'] = np.random.choice([True, False])
            config['useChirality'] = True  # Usually helpful
            
        elif model_type == 'graphconv':
            # For GraphConv, moderate complexity often works best
            config['graph_conv_layers'] = np.random.choice([[64, 64], [128, 128], [64, 128]])
            config['dense_layer_size'] = np.random.choice([128, 256])
            config['dropout'] = np.random.choice([0.2, 0.3, 0.4])
            config['learning_rate'] = np.random.choice([0.001, 0.005])
            
        elif model_type == 'chemberta':
            # For ChemBERTa, conservative settings often work
            config['max_length'] = np.random.choice([256, 512])
            config['learning_rate'] = np.random.choice([1e-5, 2e-5])
            config['batch_size'] = np.random.choice([16, 32])
            config['num_epochs'] = np.random.choice([5, 10])
        
        return config
    
    def simulate_model_training(self, model_type: str, parameters: Dict) -> Tuple[float, float]:
        """
        Simulate model training and return (performance, explanation_quality).
        In practice, this would train actual models.
        """
        
        # Simulate performance based on parameter quality
        base_performance = 0.5
        base_explanation_quality = 0.3
        
        if model_type == 'circular_fingerprint':
            # Simulate fingerprint model behavior
            if parameters['radius'] in [2, 3]:
                base_performance += 0.15
            if parameters['nBits'] >= 1024:
                base_performance += 0.1
                base_explanation_quality += 0.1
            if parameters['useFeatures']:
                base_explanation_quality += 0.15
            if parameters['useChirality']:
                base_explanation_quality += 0.1
                
        elif model_type == 'graphconv':
            # Simulate GraphConv model behavior
            layer_complexity = sum(parameters['graph_conv_layers'])
            if 64 <= layer_complexity <= 256:
                base_performance += 0.2
                base_explanation_quality += 0.15
            if parameters['dropout'] in [0.2, 0.3]:
                base_performance += 0.1
                base_explanation_quality += 0.2
            if parameters['learning_rate'] == 0.001:
                base_performance += 0.1
            if parameters['dense_layer_size'] in [128, 256]:
                base_explanation_quality += 0.1
                
        elif model_type == 'chemberta':
            # Simulate ChemBERTa model behavior  
            if parameters['max_length'] >= 256:
                base_performance += 0.15
            if parameters['learning_rate'] <= 2e-5:
                base_performance += 0.1
                base_explanation_quality += 0.15
            if parameters['batch_size'] in [16, 32]:
                base_performance += 0.05
            if parameters['num_epochs'] in [5, 10]:
                base_explanation_quality += 0.2
        
        # Add some realistic noise
        performance = base_performance + np.random.normal(0, 0.05)
        explanation_quality = base_explanation_quality + np.random.normal(0, 0.08)
        
        # Clip to valid ranges
        performance = np.clip(performance, 0.0, 1.0)
        explanation_quality = np.clip(explanation_quality, 0.0, 1.0)
        
        return performance, explanation_quality
    
    def evaluate_configuration(self, model_type: str, parameters: Dict) -> Dict:
        """Evaluate a specific parameter configuration."""
        
        print(f"🔬 Evaluating {model_type} with: {parameters}")
        
        # Simulate training (in practice, would train actual model)
        performance, explanation_quality = self.simulate_model_training(model_type, parameters)
        
        # Calculate combined score (60% performance, 40% explanation quality)
        combined_score = 0.6 * performance + 0.4 * explanation_quality
        
        result = {
            'model_type': model_type,
            'parameters': parameters,
            'performance': performance,
            'explanation_quality': explanation_quality,
            'combined_score': combined_score,
            'timestamp': datetime.now().isoformat(),
            'iteration': self.learning_state['iteration']
        }
        
        print(f"   📊 Results: Perf={performance:.3f}, Quality={explanation_quality:.3f}, Combined={combined_score:.3f}")
        
        return result
    
    def update_learning_state(self, result: Dict):
        """Update the agent's learning state based on results."""
        
        model_type = result['model_type']
        
        # Update best configuration if this is better
        if (model_type not in self.best_configurations or 
            result['combined_score'] > self.best_configurations[model_type]['combined_score']):
            
            self.best_configurations[model_type] = result
            print(f"   🏆 New best {model_type} configuration! Score: {result['combined_score']:.3f}")
            
            # Increase exploitation when finding good results
            self.learning_state['exploration_rate'] = max(0.3, self.learning_state['exploration_rate'] - 0.05)
        else:
            # Increase exploration when results are poor
            self.learning_state['exploration_rate'] = min(0.8, self.learning_state['exploration_rate'] + 0.02)
        
        # Update iteration counter
        self.learning_state['iteration'] += 1
        
        # Cool down temperature for simulated annealing effect
        self.learning_state['temperature'] *= 0.95
        
        # Add to history
        self.optimization_history.append(result)
    
    def optimize_model_type(self, model_type: str, max_iterations: int = 10) -> Dict:
        """Optimize parameters for a specific model type using agentic approach."""
        
        print(f"\n🎯 Agentic Optimization for {model_type.upper()}")
        print("=" * 50)
        
        model_results = []
        
        for iteration in range(max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # Intelligently select parameter candidates
            candidates = self.intelligent_parameter_selection(model_type, num_candidates=1)
            
            for candidate in candidates:
                # Evaluate configuration
                result = self.evaluate_configuration(model_type, candidate)
                model_results.append(result)
                
                # Update learning state
                self.update_learning_state(result)
        
        # Return summary for this model type
        best_result = self.best_configurations.get(model_type)
        
        summary = {
            'model_type': model_type,
            'iterations_completed': max_iterations,
            'best_configuration': best_result,
            'all_results': model_results,
            'improvement_over_iterations': [r['combined_score'] for r in model_results]
        }
        
        if best_result:
            print(f"\n📊 {model_type.upper()} Optimization Complete:")
            print(f"   🏆 Best Score: {best_result['combined_score']:.3f}")
            print(f"   ⚙️ Best Config: {best_result['parameters']}")
            print(f"   📈 Performance: {best_result['performance']:.3f}")
            print(f"   🔍 Explanation Quality: {best_result['explanation_quality']:.3f}")
        
        return summary
    
    def run_comprehensive_optimization(self, models_to_optimize: List[str] = None, iterations_per_model: int = 8) -> Dict:
        """Run comprehensive agentic optimization across all model types."""
        
        if models_to_optimize is None:
            models_to_optimize = ['circular_fingerprint', 'graphconv', 'chemberta']
        
        print("🚀 Starting Comprehensive Agentic Parameter Optimization")
        print("=" * 70)
        print(f"🎯 Models to optimize: {', '.join(models_to_optimize)}")
        print(f"🔄 Iterations per model: {iterations_per_model}")
        
        optimization_results = {}
        
        # Optimize each model type
        for model_type in models_to_optimize:
            try:
                result = self.optimize_model_type(model_type, iterations_per_model)
                optimization_results[model_type] = result
            except Exception as e:
                print(f"❌ Error optimizing {model_type}: {e}")
                optimization_results[model_type] = {'error': str(e)}
        
        # Find overall best model
        best_overall = None
        best_overall_score = 0
        
        for model_type, result in optimization_results.items():
            if 'best_configuration' in result and result['best_configuration']:
                score = result['best_configuration']['combined_score']
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall = {
                        'model_type': model_type,
                        'score': score,
                        'config': result['best_configuration']
                    }
        
        # Create comprehensive report
        final_report = {
            'optimization_summary': {
                'timestamp': datetime.now().isoformat(),
                'models_optimized': len(models_to_optimize),
                'total_evaluations': len(self.optimization_history),
                'best_overall_model': best_overall['model_type'] if best_overall else None,
                'best_overall_score': best_overall_score,
                'learning_state_final': self.learning_state
            },
            'model_results': optimization_results,
            'best_configurations': self.best_configurations,
            'optimization_history': self.optimization_history,
            'recommendations': self._generate_recommendations()
        }
        
        # Save results
        os.makedirs("results", exist_ok=True)
        report_path = "results/agentic_optimization_results.json"
        
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        # Print final summary
        print("\n" + "=" * 70)
        print("🏆 AGENTIC OPTIMIZATION COMPLETE")
        print("=" * 70)
        
        if best_overall:
            print(f"🥇 Best Overall Model: {best_overall['model_type'].upper()}")
            print(f"🎯 Best Score: {best_overall['score']:.3f}")
            print(f"⚙️ Best Configuration:")
            for param, value in best_overall['config']['parameters'].items():
                print(f"   • {param}: {value}")
        
        print(f"\n📊 Model Comparison:")
        for model_type, result in optimization_results.items():
            if 'best_configuration' in result and result['best_configuration']:
                score = result['best_configuration']['combined_score']
                print(f"  • {model_type:<20}: {score:.3f}")
        
        print(f"\n💾 Detailed results saved to: {report_path}")
        
        return final_report
    
    def _generate_recommendations(self) -> List[Dict]:
        """Generate intelligent recommendations based on optimization results."""
        recommendations = []
        
        # Analyze patterns across all models
        all_scores = [h['combined_score'] for h in self.optimization_history]
        
        if all_scores:
            avg_score = np.mean(all_scores)
            max_score = np.max(all_scores)
            
            if max_score < 0.7:
                recommendations.append({
                    'type': 'general',
                    'priority': 'HIGH',
                    'message': 'Overall performance is moderate. Consider expanding parameter search ranges.',
                    'action': 'Increase parameter diversity in next optimization round'
                })
            
            if avg_score < 0.5:
                recommendations.append({
                    'type': 'general',
                    'priority': 'HIGH',
                    'message': 'Average performance is low. May need fundamental model architecture changes.',
                    'action': 'Review model architectures and consider hybrid approaches'
                })
        
        # Model-specific recommendations
        for model_type, best_config in self.best_configurations.items():
            if best_config['combined_score'] < 0.6:
                recommendations.append({
                    'type': 'model_specific',
                    'model': model_type,
                    'priority': 'MEDIUM',
                    'message': f'{model_type} performance is below target. Consider parameter fine-tuning.',
                    'action': f'Focus on optimizing {model_type} parameters further'
                })
        
        return recommendations

def main():
    """Run the simple agentic parameter optimization."""
    
    print("🤖 Simple Agentic Parameter Optimizer")
    print("This demonstrates intelligent parameter exploration for better explanation quality.")
    
    # Initialize optimizer
    optimizer = SimpleAgenticOptimizer()
    
    try:
        # Run optimization
        results = optimizer.run_comprehensive_optimization(
            models_to_optimize=['circular_fingerprint', 'graphconv', 'chemberta'],
            iterations_per_model=6
        )
        
        print("\n🎉 Agentic optimization completed successfully!")
        print("The agent has learned and identified optimal parameter configurations.")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
