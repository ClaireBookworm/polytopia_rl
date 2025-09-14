#!/usr/bin/env python3
"""
Quick evaluation script to test and compare our actual Polytopia RL models

This script runs short training sessions with different approaches and measures:
1. Score improvement over time
2. Action selection patterns 
3. Learning stability
4. Sample efficiency

Usage: python quick_eval.py
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# Add repo root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

def evaluate_baseline_ppo():
    """Evaluate baseline PPO (standard CleanRL implementation)"""
    print("🔍 Evaluating Baseline PPO...")
    
    try:
        # Run baseline PPO for short training
        import subprocess
        result = subprocess.run([
            sys.executable, 
            "py_rl/cleanrl/cleanrl/ppo.py",
            "--total-timesteps", "5000",
            "--num-steps", "32", 
            "--track", "False"
        ], cwd=repo_root, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Parse output for scores
            scores = parse_scores_from_output(result.stdout)
            return {
                'name': 'Baseline PPO',
                'success': True,
                'scores': scores,
                'final_score': scores[-1] if scores else 0,
                'output': result.stdout
            }
        else:
            return {
                'name': 'Baseline PPO', 
                'success': False, 
                'error': result.stderr
            }
            
    except Exception as e:
        return {'name': 'Baseline PPO', 'success': False, 'error': str(e)}

def evaluate_action_quality_ppo():
    """Evaluate Action Quality PPO"""
    print("🔍 Evaluating Action Quality PPO...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable,
            "py_rl/cleanrl/cleanrl/ppo_action_quality.py", 
            "--total-timesteps", "5000",
            "--num-steps", "32",
            "--track", "False"
        ], cwd=repo_root, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            scores = parse_scores_from_output(result.stdout)
            return {
                'name': 'Action Quality PPO',
                'success': True, 
                'scores': scores,
                'final_score': scores[-1] if scores else 0,
                'output': result.stdout
            }
        else:
            return {
                'name': 'Action Quality PPO',
                'success': False,
                'error': result.stderr
            }
            
    except Exception as e:
        return {'name': 'Action Quality PPO', 'success': False, 'error': str(e)}

def evaluate_semantic_ppo():
    """Evaluate Semantic PPO with action type embeddings"""
    print("🔍 Evaluating Semantic PPO...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable,
            "py_rl/cleanrl/cleanrl/ppo_semantic.py",
            "--total-timesteps", "5000", 
            "--num-steps", "32",
            "--track", "False"
        ], cwd=repo_root, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            scores = parse_scores_from_output(result.stdout)
            return {
                'name': 'Semantic PPO',
                'success': True,
                'scores': scores, 
                'final_score': scores[-1] if scores else 0,
                'output': result.stdout
            }
        else:
            return {
                'name': 'Semantic PPO',
                'success': False,
                'error': result.stderr
            }
            
    except Exception as e:
        return {'name': 'Semantic PPO', 'success': False, 'error': str(e)}

def parse_scores_from_output(output: str) -> list:
    """Extract scores from training output"""
    scores = []
    lines = output.split('\n')
    
    for line in lines:
        # Look for score patterns in the output
        if 'episodic_return=' in line:
            try:
                # Extract the score value
                start = line.find('episodic_return=') + len('episodic_return=')
                end = line.find(',', start) if ',' in line[start:] else len(line)
                score_str = line[start:end].strip()
                score = float(score_str)
                scores.append(score)
            except:
                continue
                
        # Also look for score patterns from our custom output  
        elif '] [' in line and len(line.split()) >= 2:
            try:
                # Pattern like "923 [624, 623, 726]"
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    score = float(parts[0])
                    scores.append(score)
            except:
                continue
    
    return scores

def analyze_action_patterns(output: str) -> dict:
    """Analyze action selection patterns from output"""
    action_stats = defaultdict(int)
    
    lines = output.split('\n')
    for line in lines:
        # Look for action-related output
        if 'RESEARCH' in line:
            action_stats['RESEARCH'] += 1
        elif 'MOVE' in line:
            action_stats['MOVE'] += 1
        elif 'ATTACK' in line:
            action_stats['ATTACK'] += 1
        elif 'BUILD' in line:
            action_stats['BUILD'] += 1
        elif 'END_TURN' in line:
            action_stats['END_TURN'] += 1
    
    return dict(action_stats)

def calculate_learning_metrics(scores: list) -> dict:
    """Calculate learning metrics from score progression"""
    if len(scores) < 2:
        return {'learning_rate': 0, 'stability': 0, 'improvement': 0}
    
    # Learning rate (slope of improvement)
    x = np.arange(len(scores))
    y = np.array(scores)
    learning_rate = np.polyfit(x, y, 1)[0] if len(scores) > 1 else 0
    
    # Stability (inverse of variance)
    stability = 1.0 / (np.var(scores) + 1e-6)
    
    # Total improvement 
    improvement = scores[-1] - scores[0] if len(scores) > 0 else 0
    
    return {
        'learning_rate': learning_rate,
        'stability': stability, 
        'improvement': improvement
    }

def run_comprehensive_evaluation():
    """Run evaluation of all available models"""
    
    print("🚀 Quick Polytopia RL Model Evaluation")
    print("=" * 60)
    print("Testing models with 5K timesteps each...")
    print()
    
    # List of evaluation functions
    evaluations = [
        evaluate_baseline_ppo,
        evaluate_action_quality_ppo,
        evaluate_semantic_ppo,
    ]
    
    results = []
    start_time = time.time()
    
    # Run each evaluation
    for eval_func in evaluations:
        try:
            result = eval_func()
            results.append(result)
            
            if result['success']:
                print(f"✅ {result['name']}: Final score = {result['final_score']:.1f}")
            else:
                print(f"❌ {result['name']}: Failed - {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"❌ {eval_func.__name__}: Exception - {str(e)}")
            results.append({
                'name': eval_func.__name__.replace('evaluate_', '').replace('_', ' ').title(),
                'success': False,
                'error': str(e)
            })
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total evaluation time: {total_time:.1f} seconds")
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("\n❌ No models completed successfully!")
        return results
    
    print(f"\n📊 RESULTS ANALYSIS")
    print("=" * 40)
    
    # Sort by final score
    successful_results.sort(key=lambda x: x['final_score'], reverse=True)
    
    for i, result in enumerate(successful_results, 1):
        print(f"\n{i}. {result['name']}")
        print(f"   Final Score: {result['final_score']:.1f}")
        
        if 'scores' in result and result['scores']:
            metrics = calculate_learning_metrics(result['scores'])
            print(f"   Learning Rate: {metrics['learning_rate']:.3f}")
            print(f"   Improvement: {metrics['improvement']:.1f}")
            print(f"   Stability: {metrics['stability']:.2f}")
        
        # Action pattern analysis
        if 'output' in result:
            action_patterns = analyze_action_patterns(result['output'])
            if action_patterns:
                print(f"   Action Patterns: {action_patterns}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS")
    print("=" * 30)
    
    if len(successful_results) > 0:
        best = successful_results[0]
        print(f"🏆 Best Performer: {best['name']}")
        print(f"   Achieved score: {best['final_score']:.1f}")
        
        if len(successful_results) > 1:
            baseline_score = next((r['final_score'] for r in successful_results if 'Baseline' in r['name']), None)
            if baseline_score:
                improvements = []
                for result in successful_results:
                    if 'Baseline' not in result['name']:
                        improvement = (result['final_score'] - baseline_score) / baseline_score * 100
                        improvements.append((result['name'], improvement))
                
                if improvements:
                    print(f"\n📈 Improvements over baseline:")
                    for name, improvement in improvements:
                        print(f"   {name}: {improvement:+.1f}%")
    
    # Plot results if we have data
    plot_results(successful_results)
    
    return results

def plot_results(results: list):
    """Plot comparison of model performance"""
    
    if len(results) < 2:
        print("Not enough successful results to plot.")
        return
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Final Scores
        names = [r['name'] for r in results]
        scores = [r['final_score'] for r in results]
        
        bars = ax1.bar(range(len(names)), scores, alpha=0.7)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Final Score')
        ax1.set_title('Final Performance Comparison')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels([n.replace(' ', '\n') for n in names], rotation=0)
        
        # Highlight best performer
        if scores:
            best_idx = np.argmax(scores)
            bars[best_idx].set_color('gold')
        
        # Plot 2: Learning Curves
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Score')
        ax2.set_title('Learning Curves')
        
        for result in results:
            if 'scores' in result and result['scores']:
                scores = result['scores']
                # Estimate timesteps (assuming even spacing)
                timesteps = np.linspace(0, 5000, len(scores))
                ax2.plot(timesteps, scores, label=result['name'], linewidth=2, marker='o')
        
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"quick_eval_results_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved as: {filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"Could not create plots: {e}")

def main():
    """Main evaluation function"""
    return run_comprehensive_evaluation()

if __name__ == "__main__":
    results = main()
