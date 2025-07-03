#!/usr/bin/env python3
"""
Configuration Management System for GoodHunt v3+
================================================
Handles all system configuration, validation, and dynamic updates
"""

import yaml
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger('GoodHunt.Config')

@dataclass
class SystemConfig:
    """System configuration parameters"""
    max_memory_usage: int = 80
    max_cpu_usage: int = 90
    log_level: str = "INFO"
    auto_restart: bool = True
    monitoring_interval: int = 30
    max_log_files: int = 30
    
@dataclass 
class TradingConfig:
    """Trading configuration parameters"""
    max_positions: int = 10
    max_daily_loss: float = 0.05
    risk_limit: float = 0.02
    default_position_size: float = 0.1
    max_leverage: float = 1.0
    slippage_model: str = "dynamic"
    fee_model: str = "percentage"
    fee_rate: float = 0.001

@dataclass
class FeaturesConfig:
    """Features configuration"""
    enable_all_categories: bool = True
    live_trading: bool = False
    dashboard: bool = True
    alerts: bool = True
    advanced_analytics: bool = True
    risk_monitoring: bool = True
    performance_tracking: bool = True

@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    database: str = "goodhunt"
    username: str = ""
    password: str = ""
    
@dataclass
class BrokerConfig:
    """Broker configuration for live trading"""
    name: str = "paper"
    api_key: str = ""
    api_secret: str = ""
    sandbox: bool = True
    base_url: str = ""

class ConfigManager:
    """Advanced configuration manager with validation and dynamic updates"""
    
    def __init__(self, config_path: str = "utils/config.yaml"):
        self.config_path = config_path
        self.config = {}
        self.watchers = []
        self.last_modified = None
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file with validation"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Config file not found: {self.config_path}, creating default")
                self.create_default_config()
                
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                else:
                    self.config = json.load(f)
                    
            # Validate configuration
            self.validate_config()
            
            # Update last modified time
            self.last_modified = os.path.getmtime(self.config_path)
            
            logger.info(f"✅ Configuration loaded from {self.config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            return self.get_default_config()
    
    def save_config(self, config: Dict[str, Any] = None) -> bool:
        """Save configuration to file"""
        try:
            config_to_save = config or self.config
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(config_to_save, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_to_save, f, indent=2)
                    
            logger.info(f"✅ Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save config: {e}")
            return False
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "system": asdict(SystemConfig()),
            "trading": asdict(TradingConfig()),
            "features": asdict(FeaturesConfig()),
            "database": asdict(DatabaseConfig()),
            "broker": asdict(BrokerConfig()),
            "categories": {
                "A_observations": {
                    "enabled": True,
                    "features": [
                        "microstructure_indicators",
                        "sentiment_analysis", 
                        "liquidity_metrics",
                        "intermarket_signals",
                        "options_flow",
                        "economic_indicators",
                        "sector_rotation",
                        "temporal_patterns",
                        "volume_profile",
                        "fractal_analysis",
                        "mtf_confluence",
                        "momentum_persistence",
                        "efficiency_metrics",
                        "volatility_surface",
                        "regime_detection"
                    ]
                },
                "B_risk": {
                    "enabled": True,
                    "features": [
                        "dynamic_position_sizing",
                        "risk_budget_allocation",
                        "correlation_risk",
                        "tail_risk_hedging",
                        "stress_testing",
                        "var_calculation",
                        "kelly_criterion",
                        "risk_parity",
                        "cvar_calculation",
                        "dynamic_hedging",
                        "concentration_limits",
                        "liquidity_risk",
                        "risk_factor_decomposition",
                        "regime_risk_scaling",
                        "risk_dashboard"
                    ]
                },
                "C_execution": {
                    "enabled": True,
                    "features": [
                        "smart_order_routing",
                        "twap_vwap_execution",
                        "implementation_shortfall",
                        "iceberg_orders",
                        "market_impact_modeling",
                        "latency_optimization",
                        "partial_fill_handling",
                        "cross_venue_arbitrage",
                        "adaptive_execution",
                        "post_trade_analytics",
                        "pre_trade_risk_checks",
                        "order_book_analysis",
                        "execution_algorithms",
                        "trade_reporting",
                        "execution_cost_analysis"
                    ]
                },
                "D_reward": {
                    "enabled": True,
                    "features": [
                        "multi_objective_rewards",
                        "risk_adjusted_performance",
                        "benchmark_relative",
                        "regime_aware_rewards",
                        "tail_risk_penalties",
                        "consistency_bonuses",
                        "transaction_cost_integration",
                        "time_decay_rewards",
                        "diversification_bonuses",
                        "momentum_reversal_balance",
                        "volatility_adjusted_returns",
                        "mae_penalty",
                        "win_rate_optimization",
                        "skewness_preferences",
                        "dynamic_reward_scaling"
                    ]
                },
                "E_analytics": {
                    "enabled": True,
                    "features": [
                        "realtime_dashboard",
                        "trade_attribution",
                        "risk_decomposition",
                        "portfolio_optimization",
                        "backtesting_framework",
                        "monte_carlo_simulations",
                        "factor_exposure",
                        "persistence_tests",
                        "regime_performance",
                        "risk_return_plots",
                        "rolling_metrics",
                        "correlation_heatmaps",
                        "drawdown_analysis",
                        "pattern_recognition",
                        "attribution_dashboard"
                    ]
                },
                "F_adaptive": {
                    "enabled": True,
                    "features": [
                        "hyperparameter_optimization",
                        "online_learning",
                        "ensemble_management",
                        "meta_learning",
                        "transfer_learning",
                        "adversarial_training",
                        "active_learning",
                        "model_compression",
                        "continual_learning",
                        "multitask_learning",
                        "neural_architecture_search",
                        "federated_learning",
                        "curriculum_learning",
                        "self_supervised_learning",
                        "model_interpretability"
                    ]
                },
                "G_hybrid": {
                    "enabled": True,
                    "features": [
                        "expert_system_integration",
                        "technical_analysis_filters",
                        "fundamental_overlay",
                        "quantitative_factors",
                        "event_driven_strategies",
                        "seasonal_patterns",
                        "volatility_strategies",
                        "mean_reversion_filters",
                        "momentum_filters",
                        "arbitrage_scanner",
                        "risk_sentiment_detector",
                        "sector_rotation_strategies",
                        "pairs_trading",
                        "options_strategies",
                        "credit_analysis"
                    ]
                },
                "H_portfolio": {
                    "enabled": True,
                    "features": [
                        "multi_asset_allocation",
                        "dynamic_rebalancing",
                        "currency_hedging",
                        "leverage_management",
                        "tax_optimization",
                        "esg_integration",
                        "alternative_data",
                        "cross_market_arbitrage",
                        "volatility_targeting",
                        "carry_strategies",
                        "commodity_analysis",
                        "reit_strategies",
                        "fixed_income",
                        "crypto_integration",
                        "portfolio_insurance"
                    ]
                },
                "I_live_trading": {
                    "enabled": False,  # Disabled by default for safety
                    "features": [
                        "broker_api_integration",
                        "realtime_data_feeds",
                        "order_management",
                        "position_reconciliation",
                        "risk_limit_monitoring",
                        "performance_tracking",
                        "alert_management",
                        "failover_recovery",
                        "compliance_monitoring",
                        "trade_confirmation",
                        "market_hours_management",
                        "connection_monitoring",
                        "emergency_stop",
                        "latency_measurement",
                        "audit_trail"
                    ]
                },
                "J_research": {
                    "enabled": True,
                    "features": [
                        "shap_explainability",
                        "feature_importance",
                        "model_diagnostics",
                        "ab_testing",
                        "walk_forward_analysis",
                        "sensitivity_analysis",
                        "regime_analysis",
                        "factor_research",
                        "backtest_validation",
                        "significance_tests",
                        "risk_model_validation",
                        "performance_attribution",
                        "microstructure_analysis",
                        "behavioral_analysis",
                        "alternative_data_research"
                    ]
                }
            },
            "alerts": {
                "email": {
                    "enabled": False,
                    "smtp_server": "",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "from_email": "",
                    "to_emails": []
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": "",
                    "channel": "#trading"
                },
                "discord": {
                    "enabled": False,
                    "webhook_url": ""
                }
            },
            "logging": {
                "level": "INFO",
                "max_files": 30,
                "max_size_mb": 100,
                "rotate_daily": True,
                "compress_old": True
            }
        }
    
    def create_default_config(self):
        """Create default configuration file"""
        default_config = self.get_default_config()
        self.save_config(default_config)
        self.config = default_config
    
    def validate_config(self) -> bool:
        """Validate configuration parameters"""
        try:
            # Validate system config
            system = self.config.get('system', {})
            if system.get('max_memory_usage', 0) > 95:
                logger.warning("⚠️  Max memory usage > 95% may cause system instability")
                
            if system.get('max_cpu_usage', 0) > 95:
                logger.warning("⚠️  Max CPU usage > 95% may cause system instability")
            
            # Validate trading config
            trading = self.config.get('trading', {})
            if trading.get('max_daily_loss', 0) > 0.2:
                logger.warning("⚠️  Max daily loss > 20% is very risky")
                
            if trading.get('risk_limit', 0) > 0.1:
                logger.warning("⚠️  Risk limit > 10% per trade is very risky")
            
            # Validate live trading config
            features = self.config.get('features', {})
            if features.get('live_trading', False):
                broker = self.config.get('broker', {})
                if not broker.get('api_key') or not broker.get('api_secret'):
                    logger.error("❌ Live trading enabled but broker credentials missing")
                    return False
            
            logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value with dot notation support"""
        try:
            keys = key.split('.')
            config = self.config
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value
            return True
        except Exception as e:
            logger.error(f"❌ Failed to set config value {key}: {e}")
            return False
    
    def update_feature_status(self, category: str, feature: str, enabled: bool) -> bool:
        """Update feature enable/disable status"""
        try:
            feature_path = f"categories.{category}.enabled"
            current_status = self.get(feature_path, True)
            
            if current_status != enabled:
                self.set(feature_path, enabled)
                logger.info(f"{'✅ Enabled' if enabled else '❌ Disabled'} {category}.{feature}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to update feature status: {e}")
            return False
    
    def get_enabled_features(self) -> Dict[str, list]:
        """Get all enabled features by category"""
        enabled_features = {}
        categories = self.config.get('categories', {})
        
        for category, config in categories.items():
            if config.get('enabled', False):
                enabled_features[category] = config.get('features', [])
                
        return enabled_features
    
    def check_file_changes(self) -> bool:
        """Check if config file has been modified"""
        try:
            if not os.path.exists(self.config_path):
                return False
                
            current_modified = os.path.getmtime(self.config_path)
            if self.last_modified and current_modified > self.last_modified:
                logger.info("📝 Configuration file changed, reloading...")
                self.load_config()
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error checking file changes: {e}")
            return False

# Global config manager instance
config_manager = ConfigManager()

def load_config(config_path: str = "utils/config.yaml") -> Dict[str, Any]:
    """Load configuration from file"""
    global config_manager
    config_manager.config_path = config_path
    return config_manager.load_config()

def save_config(config: Dict[str, Any], config_path: str = "utils/config.yaml") -> bool:
    """Save configuration to file"""
    global config_manager
    config_manager.config_path = config_path
    return config_manager.save_config(config)

def get_config(key: str = None, default: Any = None) -> Any:
    """Get configuration value"""
    global config_manager
    if key is None:
        return config_manager.config
    return config_manager.get(key, default)

def set_config(key: str, value: Any) -> bool:
    """Set configuration value"""
    global config_manager
    return config_manager.set(key, value)

def get_enabled_features() -> Dict[str, list]:
    """Get all enabled features"""
    global config_manager
    return config_manager.get_enabled_features()