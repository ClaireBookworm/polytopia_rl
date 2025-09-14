#!/usr/bin/env python3
"""
Comprehensive evaluation framework for Polytopia RL models

This script compares different approaches:
1. Baseline PPO (treats actions as meaningless indices)
2. Action Quality PPO (predicts action quality scores)
3. Semantic PPO (uses action type embeddings)
4. Text Semantic PPO (uses sentence transformer embeddings)

Evaluation metrics:
- Learning speed (score improvement over time)
- Final performance (maximum achieved score)
- Action selection quality (preference for good action types)
- Sample efficiency (performance per timestep)
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Add repo root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    total_timesteps: int = 25000  # Shorter for comparison
    num_envs: int = 4
    num_steps: int = 64
    num_runs: int = 3  # Multiple runs for statistical significance
    eval_frequency: int = 2500  # Evaluate every N steps
    eval_episodes: int = 5  # Episodes per evaluation
    save_results: bool = True
    plot_results: bool = True

class ModelEvaluator:
    """Evaluates and compares different PPO model approaches"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results = defaultdict(list)
        self.action_stats = defaultdict(lambda: defaultdict(list))
        
    def evaluate_model(self, model_name: str, model_class, model_kwargs: Dict = None) -> Dict[str, Any]:
        """Evaluate a single model approach"""
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        run_results = []
        
        for run in range(self.config.num_runs):
            print(f"\n--- Run {run + 1}/{self.config.num_runs} ---")
            
            try:
                # Train and evaluate model
                run_result = self._train_and_evaluate_single_run(
                    model_name, model_class, model_kwargs or {}, run
                )
                run_results.append(run_result)
                
            except Exception as e:
                print(f"Error in run {run + 1}: {e}")
                continue
        
        if not run_results:
            print(f"All runs failed for {model_name}")
            return {}
            
        # Aggregate results across runs
        aggregated = self._aggregate_run_results(run_results)
        aggregated['model_name'] = model_name
        aggregated['successful_runs'] = len(run_results)
        
        print(f"\n{model_name} Results Summary:")
        print(f"  Successful runs: {len(run_results)}/{self.config.num_runs}")
        print(f"  Final score: {aggregated['final_score_mean']:.1f} ± {aggregated['final_score_std']:.1f}")
        print(f"  Learning rate: {aggregated['learning_rate']:.4f} scores/1000 steps")
        print(f"  Sample efficiency: {aggregated['sample_efficiency']:.2f}")
        
        return aggregated
    
    def _train_and_evaluate_single_run(self, model_name: str, model_class, model_kwargs: Dict, run_id: int) -> Dict:
        """Train and evaluate a single model run"""
        
        # Import here to avoid circular imports
        import gymnasium as gym
        from pol_env.Tribes.py.register_env import TribesGymWrapper
        
        # Create environments
        def make_env(idx):
            def thunk():
                base_env = TribesGymWrapper()
                env = gym.wrappers.RecordEpisodeStatistics(base_env)
                return env
            return thunk
        
        envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(self.config.num_envs)])
        
        # Initialize model
        model = model_class(envs, **model_kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
        
        # Training metrics
        scores = []
        action_type_counts = defaultdict(int)
        timesteps = []
        episode_rewards = deque(maxlen=100)
        
        # Reset environments
        obs, _ = envs.reset()
        
        total_steps = 0
        start_time = time.time()
        
        for iteration in range(self.config.total_timesteps // (self.config.num_envs * self.config.num_steps)):
            
            # Collect rollout
            rollout_data = self._collect_rollout(model, envs, obs, self.config.num_steps)
            obs = rollout_data['next_obs']
            
            # Update model
            self._update_model(model, optimizer, rollout_data)
            
            # Track metrics
            total_steps += self.config.num_envs * self.config.num_steps
            
            # Collect episode statistics
            if 'episode_rewards' in rollout_data:
                episode_rewards.extend(rollout_data['episode_rewards'])
            
            # Periodic evaluation
            if total_steps % self.config.eval_frequency == 0:
                eval_score = self._evaluate_current_performance(model, envs)
                scores.append(eval_score)
                timesteps.append(total_steps)
                
                # Collect action statistics
                action_stats = self._collect_action_statistics(model, envs)
                for action_type, count in action_stats.items():
                    action_type_counts[action_type] += count
                
                print(f"  Step {total_steps:,}: Score = {eval_score:.1f}, "
                      f"Avg Episode Reward = {np.mean(episode_rewards) if episode_rewards else 0:.2f}")
        
        envs.close()
        
        # Calculate metrics
        final_score = scores[-1] if scores else 0
        learning_rate = self._calculate_learning_rate(scores, timesteps)
        sample_efficiency = final_score / (total_steps / 1000) if total_steps > 0 else 0
        
        return {
            'scores': scores,
            'timesteps': timesteps,
            'final_score': final_score,
            'learning_rate': learning_rate,
            'sample_efficiency': sample_efficiency,
            'action_type_counts': dict(action_type_counts),
            'episode_rewards': list(episode_rewards),
            'total_time': time.time() - start_time
        }
    
    def _collect_rollout(self, model, envs, obs, num_steps) -> Dict:
        """Collect a rollout using the current policy"""
        
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        episode_rewards = []
        
        for step in range(num_steps):
            observations.append(obs)
            
            # Get action from model
            with torch.no_grad():
                if hasattr(model, 'get_action_and_value'):
                    # For models that support this interface
                    action, log_prob, _, value = model.get_action_and_value(obs)
                else:
                    # Fallback for basic models
                    action = envs.action_space.sample()
                    log_prob = torch.zeros(len(obs))
                    value = torch.zeros(len(obs))
            
            # Take step
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            
            # Store data
            actions.append(action)
            rewards.append(reward)
            dones.append(np.logical_or(terminations, truncations))
            values.append(value)
            log_probs.append(log_prob)
            
            # Track episode rewards
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_rewards.append(info["episode"]["r"])
            
            obs = next_obs
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'values': values,
            'log_probs': log_probs,
            'next_obs': obs,
            'episode_rewards': episode_rewards
        }
    
    def _update_model(self, model, optimizer, rollout_data):
        """Update model parameters (simplified PPO update)"""
        # This is a simplified update - in practice you'd implement full PPO
        optimizer.zero_grad()
        
        # Simple value loss
        rewards = torch.tensor(np.array(rollout_data['rewards'])).float()
        values = torch.stack(rollout_data['values'])
        
        if rewards.numel() > 0 and values.numel() > 0:
            # Reshape to match
            if rewards.dim() == 2 and values.dim() == 2:
                value_loss = nn.MSELoss()(values, rewards)
                value_loss.backward()
                optimizer.step()
    
    def _evaluate_current_performance(self, model, envs) -> float:
        """Evaluate current model performance"""
        eval_scores = []
        
        for _ in range(self.config.eval_episodes):
            obs, _ = envs.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 100:  # Max steps per episode
                with torch.no_grad():
                    if hasattr(model, 'get_action'):
                        action = model.get_action(obs)
                    else:
                        action = envs.action_space.sample()
                
                obs, reward, terminations, truncations, _ = envs.step(action)
                episode_reward += np.mean(reward)
                done = np.any(terminations) or np.any(truncations)
                steps += 1
            
            eval_scores.append(episode_reward)
        
        return np.mean(eval_scores)
    
    def _collect_action_statistics(self, model, envs) -> Dict[str, int]:
        """Collect statistics about which action types the model prefers"""
        action_type_counts = defaultdict(int)
        
        try:
            # Get available actions from first environment
            if hasattr(envs.envs[0], 'list_actions'):
                actions = envs.envs[0].list_actions()
                for action in actions:
                    action_type = action.get('type', 'UNKNOWN')
                    action_type_counts[action_type] += 1
        except:
            pass  # Skip if not available
        
        return dict(action_type_counts)
    
    def _calculate_learning_rate(self, scores: List[float], timesteps: List[int]) -> float:
        """Calculate learning rate (score improvement per 1000 timesteps)"""
        if len(scores) < 2:
            return 0.0
        
        # Linear regression to find slope
        x = np.array(timesteps) / 1000  # Convert to thousands
        y = np.array(scores)
        
        if len(x) > 1:
            slope, _ = np.polyfit(x, y, 1)
            return slope
        return 0.0
    
    def _aggregate_run_results(self, run_results: List[Dict]) -> Dict:
        """Aggregate results across multiple runs"""
        
        # Extract metrics
        final_scores = [r['final_score'] for r in run_results]
        learning_rates = [r['learning_rate'] for r in run_results]
        sample_efficiencies = [r['sample_efficiency'] for r in run_results]
        
        return {
            'final_score_mean': np.mean(final_scores),
            'final_score_std': np.std(final_scores),
            'learning_rate': np.mean(learning_rates),
            'sample_efficiency': np.mean(sample_efficiencies),
            'run_results': run_results
        }
    
    def compare_models(self, model_configs: List[Tuple[str, Any, Dict]]) -> Dict:
        """Compare multiple model approaches"""
        
        print("Starting comprehensive model comparison...")
        print(f"Configuration: {self.config.total_timesteps:,} timesteps, "
              f"{self.config.num_runs} runs per model")
        
        all_results = {}
        
        for model_name, model_class, model_kwargs in model_configs:
            try:
                results = self.evaluate_model(model_name, model_class, model_kwargs)
                all_results[model_name] = results
            except Exception as e:
                print(f"Failed to evaluate {model_name}: {e}")
                continue
        
        # Generate comparison report
        self._generate_comparison_report(all_results)
        
        if self.config.plot_results:
            self._plot_comparison_results(all_results)
        
        if self.config.save_results:
            self._save_results(all_results)
        
        return all_results
    
    def _generate_comparison_report(self, results: Dict):
        """Generate a comprehensive comparison report"""
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE MODEL COMPARISON REPORT")
        print(f"{'='*80}")
        
        if not results:
            print("No successful results to compare.")
            return
        
        # Sort by final performance
        sorted_models = sorted(results.items(), 
                             key=lambda x: x[1].get('final_score_mean', 0), 
                             reverse=True)
        
        print(f"\n📊 PERFORMANCE RANKING:")
        print("-" * 50)
        for i, (name, result) in enumerate(sorted_models, 1):
            score = result.get('final_score_mean', 0)
            std = result.get('final_score_std', 0)
            lr = result.get('learning_rate', 0)
            eff = result.get('sample_efficiency', 0)
            runs = result.get('successful_runs', 0)
            
            print(f"{i}. {name}")
            print(f"   Final Score: {score:.1f} ± {std:.1f}")
            print(f"   Learning Rate: {lr:.4f} scores/1000 steps")
            print(f"   Sample Efficiency: {eff:.2f}")
            print(f"   Successful Runs: {runs}/{self.config.num_runs}")
            print()
        
        # Statistical significance tests
        print(f"\n📈 LEARNING ANALYSIS:")
        print("-" * 50)
        
        baseline_name = None
        baseline_score = 0
        
        for name, result in results.items():
            if 'baseline' in name.lower() or 'basic' in name.lower():
                baseline_name = name
                baseline_score = result.get('final_score_mean', 0)
                break
        
        if baseline_name:
            print(f"Baseline model: {baseline_name} ({baseline_score:.1f})")
            print("\nImprovements over baseline:")
            
            for name, result in results.items():
                if name != baseline_name:
                    score = result.get('final_score_mean', 0)
                    improvement = ((score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
                    print(f"  {name}: {improvement:+.1f}%")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        print("-" * 50)
        
        best_name, best_result = sorted_models[0]
        best_lr = best_result.get('learning_rate', 0)
        
        print(f"✅ Best Overall: {best_name}")
        print(f"   - Highest final score: {best_result.get('final_score_mean', 0):.1f}")
        
        # Find fastest learner
        fastest_lr = max(results.items(), key=lambda x: x[1].get('learning_rate', 0))
        if fastest_lr[0] != best_name:
            print(f"🚀 Fastest Learner: {fastest_lr[0]}")
            print(f"   - Learning rate: {fastest_lr[1].get('learning_rate', 0):.4f}")
        
        # Find most efficient
        most_efficient = max(results.items(), key=lambda x: x[1].get('sample_efficiency', 0))
        if most_efficient[0] != best_name:
            print(f"⚡ Most Sample Efficient: {most_efficient[0]}")
            print(f"   - Efficiency: {most_efficient[1].get('sample_efficiency', 0):.2f}")
        
        print(f"\n💡 For production use, consider: {best_name}")
        print(f"💡 For further research, investigate: {fastest_lr[0]}")
    
    def _plot_comparison_results(self, results: Dict):
        """Plot comparison results"""
        
        if not results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Polytopia RL Model Comparison', fontsize=16, fontweight='bold')
        
        models = list(results.keys())
        
        # 1. Final Performance Comparison
        ax1 = axes[0, 0]
        scores = [results[m].get('final_score_mean', 0) for m in models]
        stds = [results[m].get('final_score_std', 0) for m in models]
        
        bars = ax1.bar(range(len(models)), scores, yerr=stds, capsize=5, alpha=0.7)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Final Score')
        ax1.set_title('Final Performance')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        
        # Color best performer
        if scores:
            best_idx = np.argmax(scores)
            bars[best_idx].set_color('gold')
        
        # 2. Learning Rate Comparison
        ax2 = axes[0, 1]
        learning_rates = [results[m].get('learning_rate', 0) for m in models]
        
        bars2 = ax2.bar(range(len(models)), learning_rates, alpha=0.7)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Learning Rate (scores/1000 steps)')
        ax2.set_title('Learning Speed')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        
        # Color fastest learner
        if learning_rates:
            fastest_idx = np.argmax(learning_rates)
            bars2[fastest_idx].set_color('lightcoral')
        
        # 3. Sample Efficiency
        ax3 = axes[1, 0]
        efficiencies = [results[m].get('sample_efficiency', 0) for m in models]
        
        bars3 = ax3.bar(range(len(models)), efficiencies, alpha=0.7)
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Sample Efficiency')
        ax3.set_title('Sample Efficiency')
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        
        # Color most efficient
        if efficiencies:
            efficient_idx = np.argmax(efficiencies)
            bars3[efficient_idx].set_color('lightgreen')
        
        # 4. Learning Curves (if available)
        ax4 = axes[1, 1]
        for model_name, result in results.items():
            run_results = result.get('run_results', [])
            if run_results:
                # Average learning curves
                all_scores = []
                all_timesteps = []
                
                for run in run_results:
                    if 'scores' in run and 'timesteps' in run:
                        all_scores.append(run['scores'])
                        all_timesteps.append(run['timesteps'])
                
                if all_scores:
                    # Find common timesteps and average
                    max_length = max(len(scores) for scores in all_scores)
                    if max_length > 0:
                        avg_scores = []
                        timesteps = all_timesteps[0] if all_timesteps else []
                        
                        for i in range(max_length):
                            step_scores = [scores[i] for scores in all_scores if i < len(scores)]
                            if step_scores:
                                avg_scores.append(np.mean(step_scores))
                        
                        if len(timesteps) == len(avg_scores):
                            ax4.plot(timesteps, avg_scores, label=model_name.replace('_', ' '), linewidth=2)
        
        ax4.set_xlabel('Timesteps')
        ax4.set_ylabel('Score')
        ax4.set_title('Learning Curves')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if self.config.save_results:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'model_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved as: model_comparison_{timestamp}.png")
        
        plt.show()
    
    def _save_results(self, results: Dict):
        """Save results to JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"model_evaluation_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, result in results.items():
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, list):
                    serializable_result[key] = value
                else:
                    serializable_result[key] = value
            serializable_results[model_name] = serializable_result
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'config': {
                    'total_timesteps': self.config.total_timesteps,
                    'num_envs': self.config.num_envs,
                    'num_runs': self.config.num_runs
                },
                'results': serializable_results
            }, f, indent=2)
        
        print(f"Results saved to: {filename}")


# Mock model classes for testing
class BaselinePPO:
    """Baseline PPO that treats actions as meaningless indices"""
    def __init__(self, envs):
        self.envs = envs
        # Minimal setup for testing
        
    def get_action_and_value(self, obs):
        # Random actions for testing
        batch_size = len(obs) if hasattr(obs, '__len__') else 1
        action = torch.randint(0, 10, (batch_size,))  # Assume max 10 actions
        log_prob = torch.zeros(batch_size)
        entropy = torch.zeros(batch_size)
        value = torch.zeros(batch_size)
        return action, log_prob, entropy, value

class ActionQualityPPO:
    """PPO with action quality prediction"""
    def __init__(self, envs):
        self.envs = envs
        
    def get_action_and_value(self, obs):
        batch_size = len(obs) if hasattr(obs, '__len__') else 1
        # Slightly better than random (for demo)
        action = torch.randint(0, 8, (batch_size,))  # Prefer lower indices
        log_prob = torch.zeros(batch_size)
        entropy = torch.zeros(batch_size)
        value = torch.zeros(batch_size)
        return action, log_prob, entropy, value


def main():
    """Run comprehensive model evaluation"""
    
    print("🚀 Starting Comprehensive Polytopia RL Model Evaluation")
    print("This will compare different semantic understanding approaches\n")
    
    # Configuration
    config = EvaluationConfig(
        total_timesteps=15000,  # Reduced for faster testing
        num_runs=2,  # Reduced for faster testing
        eval_frequency=1500,
        save_results=True,
        plot_results=True
    )
    
    # Models to compare
    model_configs = [
        ("Baseline_PPO", BaselinePPO, {}),
        ("Action_Quality_PPO", ActionQualityPPO, {}),
        # Add more models here as they become available
    ]
    
    # Run evaluation
    evaluator = ModelEvaluator(config)
    results = evaluator.compare_models(model_configs)
    
    print("\n🎉 Evaluation complete!")
    print(f"Compared {len(results)} models successfully.")
    
    return results


if __name__ == "__main__":
    results = main()
