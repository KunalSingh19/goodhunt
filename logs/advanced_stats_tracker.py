"""
GoodHunt v3+ Advanced Statistics Tracker
Comprehensive real-time tracking and analytics for all 120+ features across 10 categories:
- Real-time performance monitoring
- Feature attribution analysis
- Risk metrics tracking
- Model performance analytics
- User behavior tracking
- System health monitoring
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class AdvancedStatsTracker:
    def __init__(self, db_path="logs/goodhunt_stats.db", user_id=None):
        self.db_path = db_path
        self.user_id = user_id
        Path(db_path).parent.mkdir(exist_ok=True)
        self.init_database()
        self.setup_logging()
        self.metrics_cache = {}
        self.real_time_data = {
            'trades': [],
            'performance': [],
            'features': {},
            'alerts': []
        }
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            filename='logs/stats_tracker.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AdvancedStatsTracker')
        
    def init_database(self):
        """Initialize comprehensive stats database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trading Performance Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                action TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                pnl REAL,
                pnl_percent REAL,
                duration_minutes INTEGER,
                trade_reason TEXT,
                market_regime TEXT,
                volatility REAL,
                volume REAL,
                slippage REAL,
                commission REAL,
                net_pnl REAL,
                cumulative_pnl REAL,
                drawdown REAL,
                win_rate REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                max_adverse_excursion REAL,
                max_favorable_excursion REAL
            )
        ''')
        
        # Feature Attribution Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_attribution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feature_category TEXT,
                feature_name TEXT,
                feature_value REAL,
                feature_importance REAL,
                signal_strength REAL,
                contribution_to_pnl REAL,
                prediction_accuracy REAL,
                false_positive_rate REAL,
                true_positive_rate REAL,
                usage_frequency INTEGER
            )
        ''')
        
        # Risk Metrics Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                portfolio_value REAL,
                total_exposure REAL,
                leverage REAL,
                var_95 REAL,
                var_99 REAL,
                cvar_95 REAL,
                cvar_99 REAL,
                beta_to_market REAL,
                correlation_breakdown TEXT,
                concentration_risk REAL,
                liquidity_risk REAL,
                tail_risk REAL,
                stress_test_results TEXT,
                margin_usage REAL,
                buying_power REAL
            )
        ''')
        
        # Model Performance Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT,
                model_version TEXT,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                auc_roc REAL,
                log_loss REAL,
                prediction_latency_ms REAL,
                feature_count INTEGER,
                training_time_minutes REAL,
                last_retrain_timestamp TIMESTAMP,
                prediction_confidence REAL,
                model_drift_score REAL,
                hyperparameters TEXT
            )
        ''')
        
        # System Health Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                network_latency REAL,
                api_response_time REAL,
                active_connections INTEGER,
                error_rate REAL,
                uptime_seconds INTEGER,
                data_feed_status TEXT,
                broker_connection_status TEXT,
                alert_count INTEGER,
                backup_status TEXT
            )
        ''')
        
        # User Activity Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                activity_type TEXT,
                activity_details TEXT,
                session_duration_minutes INTEGER,
                page_views INTEGER,
                api_calls INTEGER,
                features_used TEXT,
                settings_changed TEXT,
                alerts_generated INTEGER,
                errors_encountered INTEGER
            )
        ''')
        
        # Market Data Quality Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                data_source TEXT,
                completeness_score REAL,
                accuracy_score REAL,
                timeliness_score REAL,
                missing_values INTEGER,
                outliers_detected INTEGER,
                data_latency_ms REAL,
                update_frequency_seconds REAL,
                quality_score REAL
            )
        ''')
        
        # Feature Performance Metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                category TEXT,
                feature_id TEXT,
                feature_name TEXT,
                computation_time_ms REAL,
                memory_usage_mb REAL,
                accuracy_score REAL,
                stability_score REAL,
                correlation_with_returns REAL,
                information_ratio REAL,
                usage_count INTEGER,
                error_count INTEGER,
                last_error_message TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def track_trade(self, trade_data: Dict):
        """Track individual trade with comprehensive metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate additional metrics
            trade_data = self._enhance_trade_data(trade_data)
            
            cursor.execute('''
                INSERT INTO trading_performance (
                    user_id, session_id, symbol, action, entry_price, exit_price,
                    quantity, pnl, pnl_percent, duration_minutes, trade_reason,
                    market_regime, volatility, volume, slippage, commission,
                    net_pnl, cumulative_pnl, drawdown, win_rate, sharpe_ratio,
                    sortino_ratio, max_adverse_excursion, max_favorable_excursion
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.user_id, trade_data.get('session_id'),
                trade_data.get('symbol'), trade_data.get('action'),
                trade_data.get('entry_price'), trade_data.get('exit_price'),
                trade_data.get('quantity'), trade_data.get('pnl'),
                trade_data.get('pnl_percent'), trade_data.get('duration_minutes'),
                trade_data.get('trade_reason'), trade_data.get('market_regime'),
                trade_data.get('volatility'), trade_data.get('volume'),
                trade_data.get('slippage'), trade_data.get('commission'),
                trade_data.get('net_pnl'), trade_data.get('cumulative_pnl'),
                trade_data.get('drawdown'), trade_data.get('win_rate'),
                trade_data.get('sharpe_ratio'), trade_data.get('sortino_ratio'),
                trade_data.get('mae'), trade_data.get('mfe')
            ))
            
            conn.commit()
            conn.close()
            
            # Add to real-time cache
            self.real_time_data['trades'].append(trade_data)
            if len(self.real_time_data['trades']) > 1000:
                self.real_time_data['trades'] = self.real_time_data['trades'][-1000:]
            
            self.logger.info(f"Trade tracked: {trade_data.get('symbol')} {trade_data.get('action')} PnL: {trade_data.get('pnl')}")
            
        except Exception as e:
            self.logger.error(f"Error tracking trade: {str(e)}")
    
    def track_feature_performance(self, feature_data: Dict):
        """Track feature performance and attribution"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO feature_attribution (
                    user_id, feature_category, feature_name, feature_value,
                    feature_importance, signal_strength, contribution_to_pnl,
                    prediction_accuracy, false_positive_rate, true_positive_rate,
                    usage_frequency
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.user_id, feature_data.get('category'),
                feature_data.get('name'), feature_data.get('value'),
                feature_data.get('importance'), feature_data.get('signal_strength'),
                feature_data.get('pnl_contribution'), feature_data.get('accuracy'),
                feature_data.get('fpr'), feature_data.get('tpr'),
                feature_data.get('usage_frequency', 1)
            ))
            
            conn.commit()
            conn.close()
            
            # Update feature cache
            category = feature_data.get('category')
            if category not in self.real_time_data['features']:
                self.real_time_data['features'][category] = {}
            
            self.real_time_data['features'][category][feature_data.get('name')] = feature_data
            
        except Exception as e:
            self.logger.error(f"Error tracking feature performance: {str(e)}")
    
    def track_risk_metrics(self, risk_data: Dict):
        """Track comprehensive risk metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_metrics (
                    user_id, portfolio_value, total_exposure, leverage,
                    var_95, var_99, cvar_95, cvar_99, beta_to_market,
                    correlation_breakdown, concentration_risk, liquidity_risk,
                    tail_risk, stress_test_results, margin_usage, buying_power
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.user_id, risk_data.get('portfolio_value'),
                risk_data.get('total_exposure'), risk_data.get('leverage'),
                risk_data.get('var_95'), risk_data.get('var_99'),
                risk_data.get('cvar_95'), risk_data.get('cvar_99'),
                risk_data.get('beta_to_market'),
                json.dumps(risk_data.get('correlation_breakdown', {})),
                risk_data.get('concentration_risk'),
                risk_data.get('liquidity_risk'), risk_data.get('tail_risk'),
                json.dumps(risk_data.get('stress_test_results', {})),
                risk_data.get('margin_usage'), risk_data.get('buying_power')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error tracking risk metrics: {str(e)}")
    
    def track_model_performance(self, model_data: Dict):
        """Track ML model performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_performance (
                    user_id, model_name, model_version, accuracy, precision_score,
                    recall, f1_score, auc_roc, log_loss, prediction_latency_ms,
                    feature_count, training_time_minutes, last_retrain_timestamp,
                    prediction_confidence, model_drift_score, hyperparameters
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.user_id, model_data.get('model_name'),
                model_data.get('model_version'), model_data.get('accuracy'),
                model_data.get('precision'), model_data.get('recall'),
                model_data.get('f1_score'), model_data.get('auc_roc'),
                model_data.get('log_loss'), model_data.get('prediction_latency_ms'),
                model_data.get('feature_count'), model_data.get('training_time_minutes'),
                model_data.get('last_retrain_timestamp'),
                model_data.get('prediction_confidence'),
                model_data.get('model_drift_score'),
                json.dumps(model_data.get('hyperparameters', {}))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error tracking model performance: {str(e)}")
    
    def track_system_health(self, health_data: Dict):
        """Track system health metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_health (
                    cpu_usage, memory_usage, disk_usage, network_latency,
                    api_response_time, active_connections, error_rate,
                    uptime_seconds, data_feed_status, broker_connection_status,
                    alert_count, backup_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                health_data.get('cpu_usage'), health_data.get('memory_usage'),
                health_data.get('disk_usage'), health_data.get('network_latency'),
                health_data.get('api_response_time'), health_data.get('active_connections'),
                health_data.get('error_rate'), health_data.get('uptime_seconds'),
                health_data.get('data_feed_status'), health_data.get('broker_connection_status'),
                health_data.get('alert_count'), health_data.get('backup_status')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error tracking system health: {str(e)}")
    
    def _enhance_trade_data(self, trade_data: Dict) -> Dict:
        """Enhance trade data with calculated metrics"""
        # Calculate additional metrics if not provided
        if 'pnl_percent' not in trade_data and 'pnl' in trade_data and 'entry_price' in trade_data:
            entry_value = trade_data.get('entry_price', 0) * trade_data.get('quantity', 0)
            if entry_value > 0:
                trade_data['pnl_percent'] = (trade_data.get('pnl', 0) / entry_value) * 100
        
        # Calculate duration if timestamps provided
        if 'entry_time' in trade_data and 'exit_time' in trade_data:
            entry_time = datetime.fromisoformat(trade_data['entry_time'])
            exit_time = datetime.fromisoformat(trade_data['exit_time'])
            duration = (exit_time - entry_time).total_seconds() / 60
            trade_data['duration_minutes'] = duration
        
        return trade_data
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """Get comprehensive performance summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get trading performance
            trades_df = pd.read_sql_query('''
                SELECT * FROM trading_performance 
                WHERE user_id = ? AND timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days), conn, params=[self.user_id])
            
            # Get feature attribution
            features_df = pd.read_sql_query('''
                SELECT * FROM feature_attribution 
                WHERE user_id = ? AND timestamp >= datetime('now', '-{} days')
            '''.format(days), conn, params=[self.user_id])
            
            # Get risk metrics
            risk_df = pd.read_sql_query('''
                SELECT * FROM risk_metrics 
                WHERE user_id = ? AND timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days), conn, params=[self.user_id])
            
            conn.close()
            
            summary = {
                'period_days': days,
                'total_trades': len(trades_df),
                'trading_performance': self._calculate_trading_metrics(trades_df),
                'feature_performance': self._calculate_feature_metrics(features_df),
                'risk_metrics': self._calculate_risk_summary(risk_df),
                'system_health': self._get_system_health_summary(),
                'generated_at': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {str(e)}")
            return {}
    
    def _calculate_trading_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive trading metrics"""
        if trades_df.empty:
            return {}
        
        return {
            'total_pnl': trades_df['pnl'].sum(),
            'total_net_pnl': trades_df['net_pnl'].sum(),
            'win_rate': (trades_df['pnl'] > 0).mean() * 100,
            'average_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean(),
            'average_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean(),
            'profit_factor': abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                               (trades_df[trades_df['pnl'] < 0]['pnl'].sum() + 1e-9)),
            'max_drawdown': trades_df['drawdown'].max(),
            'sharpe_ratio': trades_df['sharpe_ratio'].iloc[-1] if not trades_df.empty else 0,
            'sortino_ratio': trades_df['sortino_ratio'].iloc[-1] if not trades_df.empty else 0,
            'total_commission': trades_df['commission'].sum(),
            'total_slippage': trades_df['slippage'].sum(),
            'average_trade_duration': trades_df['duration_minutes'].mean(),
            'max_adverse_excursion': trades_df['max_adverse_excursion'].max(),
            'max_favorable_excursion': trades_df['max_favorable_excursion'].max()
        }
    
    def _calculate_feature_metrics(self, features_df: pd.DataFrame) -> Dict:
        """Calculate feature performance metrics"""
        if features_df.empty:
            return {}
        
        # Top performing features by category
        feature_performance = {}
        for category in features_df['feature_category'].unique():
            cat_features = features_df[features_df['feature_category'] == category]
            feature_performance[category] = {
                'average_importance': cat_features['feature_importance'].mean(),
                'average_accuracy': cat_features['prediction_accuracy'].mean(),
                'total_pnl_contribution': cat_features['contribution_to_pnl'].sum(),
                'usage_frequency': cat_features['usage_frequency'].sum(),
                'signal_strength': cat_features['signal_strength'].mean()
            }
        
        return {
            'total_features_used': len(features_df['feature_name'].unique()),
            'average_feature_importance': features_df['feature_importance'].mean(),
            'average_prediction_accuracy': features_df['prediction_accuracy'].mean(),
            'total_feature_pnl_contribution': features_df['contribution_to_pnl'].sum(),
            'category_performance': feature_performance
        }
    
    def _calculate_risk_summary(self, risk_df: pd.DataFrame) -> Dict:
        """Calculate risk summary metrics"""
        if risk_df.empty:
            return {}
        
        latest_risk = risk_df.iloc[0] if not risk_df.empty else {}
        
        return {
            'current_portfolio_value': latest_risk.get('portfolio_value', 0),
            'current_exposure': latest_risk.get('total_exposure', 0),
            'current_leverage': latest_risk.get('leverage', 0),
            'var_95': latest_risk.get('var_95', 0),
            'var_99': latest_risk.get('var_99', 0),
            'cvar_95': latest_risk.get('cvar_95', 0),
            'beta_to_market': latest_risk.get('beta_to_market', 0),
            'concentration_risk': latest_risk.get('concentration_risk', 0),
            'liquidity_risk': latest_risk.get('liquidity_risk', 0),
            'tail_risk': latest_risk.get('tail_risk', 0)
        }
    
    def _get_system_health_summary(self) -> Dict:
        """Get system health summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            health_df = pd.read_sql_query('''
                SELECT * FROM system_health 
                ORDER BY timestamp DESC LIMIT 1
            ''', conn)
            conn.close()
            
            if health_df.empty:
                return {}
            
            latest = health_df.iloc[0]
            return {
                'cpu_usage': latest.get('cpu_usage', 0),
                'memory_usage': latest.get('memory_usage', 0),
                'network_latency': latest.get('network_latency', 0),
                'api_response_time': latest.get('api_response_time', 0),
                'error_rate': latest.get('error_rate', 0),
                'uptime_hours': latest.get('uptime_seconds', 0) / 3600,
                'data_feed_status': latest.get('data_feed_status', 'unknown')
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {str(e)}")
            return {}
    
    def generate_feature_attribution_report(self) -> Dict:
        """Generate detailed feature attribution report"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get all feature data
            features_df = pd.read_sql_query('''
                SELECT feature_category, feature_name, 
                       AVG(feature_importance) as avg_importance,
                       AVG(prediction_accuracy) as avg_accuracy,
                       SUM(contribution_to_pnl) as total_pnl_contribution,
                       AVG(signal_strength) as avg_signal_strength,
                       SUM(usage_frequency) as total_usage,
                       COUNT(*) as measurement_count
                FROM feature_attribution 
                WHERE user_id = ?
                GROUP BY feature_category, feature_name
                ORDER BY total_pnl_contribution DESC
            ''', conn, params=[self.user_id])
            
            conn.close()
            
            if features_df.empty:
                return {'error': 'No feature data available'}
            
            # Top features by PnL contribution
            top_features = features_df.head(20).to_dict('records')
            
            # Category breakdown
            category_summary = features_df.groupby('feature_category').agg({
                'total_pnl_contribution': 'sum',
                'avg_importance': 'mean',
                'avg_accuracy': 'mean',
                'total_usage': 'sum'
            }).to_dict('index')
            
            return {
                'top_features': top_features,
                'category_summary': category_summary,
                'total_features': len(features_df),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating feature attribution report: {str(e)}")
            return {'error': str(e)}
    
    def create_performance_dashboard(self, output_file: str = "logs/performance_dashboard.html"):
        """Create interactive performance dashboard"""
        try:
            # Get data
            summary = self.get_performance_summary(days=30)
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('PnL Over Time', 'Feature Performance', 
                              'Risk Metrics', 'Trade Distribution',
                              'System Health', 'Win Rate Trend'),
                specs=[[{"secondary_y": True}, {"type": "bar"}],
                       [{"type": "indicator"}, {"type": "pie"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Add plots based on available data
            if 'trading_performance' in summary:
                # PnL chart
                fig.add_trace(
                    go.Scatter(x=list(range(30)), y=[0]*30, name="Cumulative PnL"),
                    row=1, col=1
                )
            
            # Feature performance
            if 'feature_performance' in summary:
                categories = list(summary['feature_performance'].get('category_performance', {}).keys())
                values = [summary['feature_performance']['category_performance'][cat]['total_pnl_contribution'] 
                         for cat in categories]
                fig.add_trace(
                    go.Bar(x=categories, y=values, name="Feature PnL Contribution"),
                    row=1, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="GoodHunt v3+ Performance Dashboard",
                height=800,
                showlegend=True
            )
            
            # Save dashboard
            fig.write_html(output_file)
            self.logger.info(f"Performance dashboard saved to {output_file}")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error creating performance dashboard: {str(e)}")
            return None
    
    def export_detailed_stats(self, output_dir: str = "logs/detailed_exports/"):
        """Export all detailed statistics to CSV files"""
        try:
            Path(output_dir).mkdir(exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            
            # Export all tables
            tables = [
                'trading_performance', 'feature_attribution', 'risk_metrics',
                'model_performance', 'system_health', 'user_activity',
                'market_data_quality', 'feature_performance'
            ]
            
            exported_files = []
            for table in tables:
                df = pd.read_sql_query(f'SELECT * FROM {table}', conn)
                if not df.empty:
                    filename = f"{output_dir}/{table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    df.to_csv(filename, index=False)
                    exported_files.append(filename)
            
            conn.close()
            
            # Create summary report
            summary_file = f"{output_dir}/summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            summary = self.get_performance_summary(days=90)
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            exported_files.append(summary_file)
            
            self.logger.info(f"Exported {len(exported_files)} detailed stats files")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Error exporting detailed stats: {str(e)}")
            return []

# Convenience functions for easy integration
def track_trade(trade_data: Dict, user_id: int = None):
    """Quick trade tracking"""
    tracker = AdvancedStatsTracker(user_id=user_id)
    tracker.track_trade(trade_data)

def track_feature(feature_data: Dict, user_id: int = None):
    """Quick feature tracking"""
    tracker = AdvancedStatsTracker(user_id=user_id)
    tracker.track_feature_performance(feature_data)

def get_stats_summary(user_id: int = None, days: int = 30) -> Dict:
    """Quick stats summary"""
    tracker = AdvancedStatsTracker(user_id=user_id)
    return tracker.get_performance_summary(days=days)