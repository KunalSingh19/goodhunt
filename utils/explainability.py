#!/usr/bin/env python3
"""
🔍 GoodHunt Explainability Engine
Advanced AI Explainability for Trading Decisions using SHAP, LIME, and Custom Methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    import torch
    import torch.nn as nn
    EXPLAINABILITY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Explainability dependencies not available: {e}")
    EXPLAINABILITY_AVAILABLE = False

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import os
from datetime import datetime
import json

@dataclass
class ExplanationResult:
    """Container for explanation results"""
    method: str
    feature_importance: Dict[str, float]
    local_explanations: Dict[int, Dict[str, float]]
    global_explanations: Dict[str, float]
    prediction_confidence: float
    explanation_quality: float
    timestamp: str

class RLExplainer:
    """Explainability system for RL trading agents"""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.explanations_history = []
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_importance_history = {}
        
    def setup_shap_explainer(self, model, background_data: np.ndarray):
        """Setup SHAP explainer for the model"""
        if not EXPLAINABILITY_AVAILABLE:
            print("⚠️  SHAP not available")
            return False
            
        try:
            # For tree-based models
            if hasattr(model, 'predict'):
                self.shap_explainer = shap.Explainer(model, background_data[:100])
            else:
                # For neural networks - use KernelExplainer
                def model_predict(X):
                    if hasattr(model, 'predict'):
                        if len(X.shape) == 1:
                            X = X.reshape(1, -1)
                        predictions = model.predict(X)
                        if hasattr(predictions, 'flatten'):
                            return predictions.flatten()
                        return predictions
                    return np.zeros(len(X))
                
                self.shap_explainer = shap.KernelExplainer(
                    model_predict, 
                    background_data[:50]  # Smaller sample for speed
                )
            
            print("✅ SHAP explainer initialized")
            return True
            
        except Exception as e:
            logging.error(f"SHAP setup failed: {e}")
            return False
    
    def setup_lime_explainer(self, training_data: np.ndarray, mode='classification'):
        """Setup LIME explainer"""
        if not EXPLAINABILITY_AVAILABLE:
            print("⚠️  LIME not available")
            return False
            
        try:
            self.lime_explainer = LimeTabularExplainer(
                training_data,
                feature_names=self.feature_names,
                class_names=['HOLD', 'BUY', 'SELL', 'SCALE_UP', 'SCALE_DOWN', 'FLAT'],
                mode=mode,
                discretize_continuous=True,
                random_state=42
            )
            
            print("✅ LIME explainer initialized")
            return True
            
        except Exception as e:
            logging.error(f"LIME setup failed: {e}")
            return False
    
    def explain_decision_shap(self, model, instance: np.ndarray, action: int) -> Dict:
        """Generate SHAP explanation for a single decision"""
        if not EXPLAINABILITY_AVAILABLE or self.shap_explainer is None:
            return {}
        
        try:
            # Ensure instance is 2D
            if len(instance.shape) == 1:
                instance = instance.reshape(1, -1)
            
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(instance)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class output
                shap_vals = shap_values[action]
                if len(shap_vals.shape) > 1:
                    shap_vals = shap_vals[0]
            else:
                # Single output
                if len(shap_values.shape) > 1:
                    shap_vals = shap_values[0]
                else:
                    shap_vals = shap_values
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature in enumerate(self.feature_names[:len(shap_vals)]):
                feature_importance[feature] = float(shap_vals[i])
            
            # Sort by absolute importance
            sorted_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            ))
            
            return {
                'method': 'SHAP',
                'feature_importance': sorted_importance,
                'raw_shap_values': shap_vals.tolist() if hasattr(shap_vals, 'tolist') else shap_vals,
                'action_explained': action,
                'top_features': list(sorted_importance.keys())[:10]
            }
            
        except Exception as e:
            logging.error(f"SHAP explanation failed: {e}")
            return {}
    
    def explain_decision_lime(self, model, instance: np.ndarray, action: int) -> Dict:
        """Generate LIME explanation for a single decision"""
        if not EXPLAINABILITY_AVAILABLE or self.lime_explainer is None:
            return {}
        
        try:
            # Define prediction function for LIME
            def predict_fn(X):
                if hasattr(model, 'predict'):
                    return model.predict(X)
                elif hasattr(model, 'predict_proba'):
                    return model.predict_proba(X)
                else:
                    # For custom models
                    return np.random.rand(len(X), 6)  # 6 actions
            
            # Flatten instance if needed
            if len(instance.shape) > 1:
                instance_flat = instance.flatten()
            else:
                instance_flat = instance
            
            # Truncate to match feature names
            instance_flat = instance_flat[:len(self.feature_names)]
            
            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                instance_flat,
                predict_fn,
                num_features=min(20, len(self.feature_names)),
                num_samples=500
            )
            
            # Extract feature importance
            feature_importance = {}
            for feature_idx, importance in explanation.as_list():
                if feature_idx < len(self.feature_names):
                    feature_name = self.feature_names[feature_idx]
                    feature_importance[feature_name] = importance
            
            return {
                'method': 'LIME',
                'feature_importance': feature_importance,
                'action_explained': action,
                'explanation_fit': explanation.score,
                'local_pred': explanation.local_pred
            }
            
        except Exception as e:
            logging.error(f"LIME explanation failed: {e}")
            return {}
    
    def explain_decision_permutation(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Generate permutation importance explanation"""
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_test, y_test, 
                n_repeats=10, 
                random_state=42,
                n_jobs=-1
            )
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature in enumerate(self.feature_names[:len(perm_importance.importances_mean)]):
                feature_importance[feature] = float(perm_importance.importances_mean[i])
            
            # Sort by importance
            sorted_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            ))
            
            return {
                'method': 'Permutation',
                'feature_importance': sorted_importance,
                'importance_std': perm_importance.importances_std.tolist(),
                'top_features': list(sorted_importance.keys())[:15]
            }
            
        except Exception as e:
            logging.error(f"Permutation importance failed: {e}")
            return {}
    
    def explain_feature_interactions(self, model, data: np.ndarray, feature_pairs: List[Tuple[str, str]] = None) -> Dict:
        """Analyze feature interactions using SHAP"""
        if not EXPLAINABILITY_AVAILABLE or self.shap_explainer is None:
            return {}
        
        try:
            # Get SHAP interaction values
            shap_interaction_values = self.shap_explainer.shap_interaction_values(data[:100])
            
            interactions = {}
            
            if feature_pairs is None:
                # Auto-select top feature pairs
                feature_pairs = []
                n_features = min(10, len(self.feature_names))
                for i in range(n_features):
                    for j in range(i+1, n_features):
                        feature_pairs.append((self.feature_names[i], self.feature_names[j]))
            
            for feat1, feat2 in feature_pairs[:20]:  # Limit to top 20 pairs
                try:
                    idx1 = self.feature_names.index(feat1)
                    idx2 = self.feature_names.index(feat2)
                    
                    if idx1 < shap_interaction_values.shape[1] and idx2 < shap_interaction_values.shape[2]:
                        interaction_strength = np.mean(np.abs(shap_interaction_values[:, idx1, idx2]))
                        interactions[f"{feat1}_x_{feat2}"] = float(interaction_strength)
                except (ValueError, IndexError):
                    continue
            
            # Sort by interaction strength
            sorted_interactions = dict(sorted(
                interactions.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            return {
                'method': 'SHAP_Interactions',
                'feature_interactions': sorted_interactions,
                'top_interactions': list(sorted_interactions.keys())[:10]
            }
            
        except Exception as e:
            logging.error(f"Feature interaction analysis failed: {e}")
            return {}
    
    def create_explanation_dashboard(self, explanations: List[Dict], save_path: str = "explanations_dashboard.html"):
        """Create interactive dashboard for explanations"""
        try:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'Feature Importance Over Time',
                    'Top Features by Method',
                    'Action Distribution',
                    'Explanation Quality',
                    'Feature Correlation',
                    'Decision Confidence'
                ],
                specs=[
                    [{"secondary_y": False}, {"type": "bar"}],
                    [{"type": "pie"}, {"type": "scatter"}],
                    [{"type": "heatmap"}, {"type": "histogram"}]
                ]
            )
            
            # Extract data for visualization
            timestamps = []
            feature_importance_over_time = {}
            methods = []
            actions = []
            
            for exp in explanations:
                timestamps.append(exp.get('timestamp', datetime.now().isoformat()))
                methods.append(exp.get('method', 'Unknown'))
                actions.append(exp.get('action_explained', 0))
                
                for feature, importance in exp.get('feature_importance', {}).items():
                    if feature not in feature_importance_over_time:
                        feature_importance_over_time[feature] = []
                    feature_importance_over_time[feature].append(importance)
            
            # Plot 1: Feature importance over time
            for feature, importances in list(feature_importance_over_time.items())[:10]:
                fig.add_trace(
                    go.Scatter(
                        x=timestamps[:len(importances)],
                        y=importances,
                        name=feature,
                        mode='lines+markers'
                    ),
                    row=1, col=1
                )
            
            # Plot 2: Top features by method
            if explanations:
                latest_exp = explanations[-1]
                features = list(latest_exp.get('feature_importance', {}).keys())[:10]
                importances = list(latest_exp.get('feature_importance', {}).values())[:10]
                
                fig.add_trace(
                    go.Bar(x=features, y=importances, name='Feature Importance'),
                    row=1, col=2
                )
            
            # Plot 3: Action distribution
            action_names = ['HOLD', 'BUY', 'SELL', 'SCALE_UP', 'SCALE_DOWN', 'FLAT']
            action_counts = [actions.count(i) for i in range(6)]
            
            fig.add_trace(
                go.Pie(labels=action_names, values=action_counts, name="Actions"),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title="🔍 GoodHunt Explainability Dashboard",
                height=1000,
                showlegend=True
            )
            
            # Save dashboard
            fig.write_html(save_path)
            print(f"✅ Explanation dashboard saved: {save_path}")
            
            return fig
            
        except Exception as e:
            logging.error(f"Dashboard creation failed: {e}")
            return None
    
    def generate_explanation_report(self, explanations: List[Dict], save_path: str = "explanation_report.md"):
        """Generate comprehensive explanation report"""
        try:
            report = f"""# 🔍 GoodHunt AI Explainability Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Explanations:** {len(explanations)}
- **Methods Used:** {set(exp.get('method', 'Unknown') for exp in explanations)}
- **Actions Analyzed:** {len(set(exp.get('action_explained', 0) for exp in explanations))}

## Top Features Analysis

"""
            
            # Aggregate feature importance across all explanations
            all_features = {}
            for exp in explanations:
                for feature, importance in exp.get('feature_importance', {}).items():
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(abs(importance))
            
            # Calculate average importance
            avg_importance = {}
            for feature, importances in all_features.items():
                avg_importance[feature] = np.mean(importances)
            
            # Sort and add to report
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            report += "### Most Important Features\n\n"
            report += "| Rank | Feature | Avg Importance | Frequency |\n"
            report += "|------|---------|----------------|----------|\n"
            
            for i, (feature, importance) in enumerate(sorted_features[:15], 1):
                frequency = len(all_features[feature])
                report += f"| {i} | {feature} | {importance:.4f} | {frequency} |\n"
            
            # Add insights
            report += f"""

## Key Insights

### 🎯 Decision Patterns
- **Most Explained Action:** {max(set(exp.get('action_explained', 0) for exp in explanations), key=lambda x: [exp.get('action_explained', 0) for exp in explanations].count(x))}
- **Average Explanation Quality:** {np.mean([exp.get('explanation_quality', 0.5) for exp in explanations]):.3f}

### 📊 Feature Categories
- **Technical Indicators:** {len([f for f in sorted_features[:10] if any(indicator in f[0].lower() for indicator in ['rsi', 'macd', 'sma', 'ema'])])} features
- **Volume Features:** {len([f for f in sorted_features[:10] if 'volume' in f[0].lower()])} features
- **Advanced Features:** {len([f for f in sorted_features[:10] if any(advanced in f[0].lower() for advanced in ['sentiment', 'regime', 'fractal'])])} features

### 🔍 Method Comparison
"""
            
            # Method comparison
            method_performance = {}
            for exp in explanations:
                method = exp.get('method', 'Unknown')
                if method not in method_performance:
                    method_performance[method] = []
                method_performance[method].append(exp.get('explanation_quality', 0.5))
            
            for method, qualities in method_performance.items():
                avg_quality = np.mean(qualities)
                report += f"- **{method}:** {avg_quality:.3f} avg quality ({len(qualities)} explanations)\n"
            
            report += f"""

## Recommendations

### 🚀 Model Improvements
1. **Focus on top {len(sorted_features[:5])} features** for enhanced performance
2. **Monitor feature stability** across different market regimes
3. **Consider feature engineering** for underperforming indicators

### 🔧 Explainability Enhancements
1. **Increase explanation frequency** for critical decisions
2. **Add ensemble explanations** combining multiple methods
3. **Implement real-time explanation monitoring**

---
*Generated by GoodHunt Explainability Engine v3.0*
"""
            
            # Save report
            with open(save_path, 'w') as f:
                f.write(report)
            
            print(f"✅ Explanation report saved: {save_path}")
            return report
            
        except Exception as e:
            logging.error(f"Report generation failed: {e}")
            return None

class TradeExplainer:
    """Specialized explainer for individual trades"""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.trade_explanations = {}
        
    def explain_trade_entry(self, model, market_state: np.ndarray, trade_data: Dict) -> Dict:
        """Explain why a trade was entered"""
        try:
            explainer = RLExplainer(self.feature_names)
            
            # Setup basic explainer
            explanation = {
                'trade_id': trade_data.get('trade_id', 'unknown'),
                'entry_price': trade_data.get('entry_price', 0),
                'action': trade_data.get('action', 'UNKNOWN'),
                'market_conditions': {},
                'feature_contributions': {},
                'confidence_score': 0.0,
                'risk_factors': []
            }
            
            # Analyze market conditions at entry
            if len(market_state) > 0:
                # Map market state to named features
                feature_values = {}
                for i, feature in enumerate(self.feature_names[:len(market_state)]):
                    feature_values[feature] = float(market_state[i])
                
                explanation['market_conditions'] = feature_values
                
                # Identify key contributors
                sorted_features = sorted(
                    feature_values.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                
                explanation['top_conditions'] = dict(sorted_features[:10])
                
                # Assess risk factors
                risk_factors = []
                
                # High volatility
                if 'volatility' in feature_values and feature_values['volatility'] > 0.02:
                    risk_factors.append(f"High volatility: {feature_values['volatility']:.4f}")
                
                # Extreme RSI
                if 'rsi' in feature_values:
                    rsi = feature_values['rsi']
                    if rsi > 80:
                        risk_factors.append(f"Overbought RSI: {rsi:.1f}")
                    elif rsi < 20:
                        risk_factors.append(f"Oversold RSI: {rsi:.1f}")
                
                # Low volume
                if 'volume' in feature_values and feature_values['volume'] < 0.5:
                    risk_factors.append(f"Low volume: {feature_values['volume']:.3f}")
                
                explanation['risk_factors'] = risk_factors
                explanation['confidence_score'] = max(0.1, 1.0 - len(risk_factors) * 0.2)
            
            return explanation
            
        except Exception as e:
            logging.error(f"Trade entry explanation failed: {e}")
            return {}
    
    def explain_trade_exit(self, trade_data: Dict, exit_conditions: Dict) -> Dict:
        """Explain why a trade was exited"""
        try:
            explanation = {
                'trade_id': trade_data.get('trade_id', 'unknown'),
                'exit_reason': exit_conditions.get('reason', 'unknown'),
                'pnl': trade_data.get('pnl', 0),
                'duration': trade_data.get('duration_minutes', 0),
                'exit_triggers': [],
                'performance_analysis': {}
            }
            
            # Analyze exit triggers
            exit_triggers = []
            
            if exit_conditions.get('stop_loss_hit', False):
                exit_triggers.append("Stop loss triggered")
            
            if exit_conditions.get('take_profit_hit', False):
                exit_triggers.append("Take profit reached")
            
            if exit_conditions.get('time_limit', False):
                exit_triggers.append("Time limit reached")
            
            if exit_conditions.get('regime_change', False):
                exit_triggers.append("Market regime changed")
            
            explanation['exit_triggers'] = exit_triggers
            
            # Performance analysis
            pnl = trade_data.get('pnl', 0)
            duration = trade_data.get('duration_minutes', 1)
            
            explanation['performance_analysis'] = {
                'profitable': pnl > 0,
                'pnl_percent': pnl / trade_data.get('entry_price', 1) * 100,
                'pnl_per_minute': pnl / max(duration, 1),
                'trade_quality': 'Excellent' if pnl > 0.02 else 'Good' if pnl > 0 else 'Poor'
            }
            
            return explanation
            
        except Exception as e:
            logging.error(f"Trade exit explanation failed: {e}")
            return {}

# Utility functions for explanation visualization
def plot_feature_importance(feature_importance: Dict, title: str = "Feature Importance", top_n: int = 15):
    """Plot feature importance"""
    try:
        # Sort and limit features
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        
        features, importances = zip(*sorted_features)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        colors = ['green' if imp > 0 else 'red' for imp in importances]
        bars = plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            plt.text(imp + max(importances) * 0.01 if imp > 0 else imp - max(importances) * 0.01, 
                    i, f'{imp:.3f}', va='center', ha='left' if imp > 0 else 'right')
        
        plt.tight_layout()
        return plt.gcf()
        
    except Exception as e:
        logging.error(f"Feature importance plotting failed: {e}")
        return None

def create_explanation_summary(explanations: List[Dict]) -> Dict:
    """Create summary statistics for explanations"""
    try:
        summary = {
            'total_explanations': len(explanations),
            'methods_used': set(),
            'avg_confidence': 0.0,
            'top_features': {},
            'action_distribution': {},
            'temporal_patterns': {}
        }
        
        if not explanations:
            return summary
        
        # Aggregate data
        all_importances = {}
        confidences = []
        actions = []
        
        for exp in explanations:
            # Methods
            summary['methods_used'].add(exp.get('method', 'Unknown'))
            
            # Confidence
            confidences.append(exp.get('explanation_quality', 0.5))
            
            # Actions
            actions.append(exp.get('action_explained', 0))
            
            # Feature importance
            for feature, importance in exp.get('feature_importance', {}).items():
                if feature not in all_importances:
                    all_importances[feature] = []
                all_importances[feature].append(abs(importance))
        
        # Calculate summary statistics
        summary['avg_confidence'] = np.mean(confidences) if confidences else 0.0
        
        # Top features by average importance
        for feature, importances in all_importances.items():
            summary['top_features'][feature] = {
                'avg_importance': np.mean(importances),
                'frequency': len(importances),
                'std_importance': np.std(importances)
            }
        
        # Action distribution
        action_names = ['HOLD', 'BUY', 'SELL', 'SCALE_UP', 'SCALE_DOWN', 'FLAT']
        for i, action_name in enumerate(action_names):
            summary['action_distribution'][action_name] = actions.count(i)
        
        return summary
        
    except Exception as e:
        logging.error(f"Summary creation failed: {e}")
        return {}

# Example usage
if __name__ == "__main__":
    # Example feature names
    feature_names = [
        'rsi', 'macd', 'volume', 'volatility', 'sentiment', 
        'regime', 'fractal_high', 'momentum_persistence'
    ]
    
    # Create explainer
    explainer = RLExplainer(feature_names)
    
    print("🔍 GoodHunt Explainability Engine Ready")
    print(f"📊 Features tracked: {len(feature_names)}")
    print("🎯 Methods available: SHAP, LIME, Permutation")