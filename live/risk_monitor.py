#!/usr/bin/env python3
"""
Real-time Risk Monitoring System for GoodHunt v3+
================================================
Advanced risk management with real-time monitoring and alerts
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque

logger = logging.getLogger('GoodHunt.RiskMonitor')

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskType(Enum):
    POSITION_SIZE = "position_size"
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    LEVERAGE = "leverage"
    LIQUIDITY = "liquidity"

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    timestamp: datetime
    risk_type: RiskType
    level: RiskLevel
    symbol: str
    current_value: float
    threshold: float
    message: str
    suggested_action: str

@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    timestamp: datetime
    total_exposure: float
    daily_pnl: float
    unrealized_pnl: float
    max_drawdown: float
    var_95: float
    sharpe_ratio: float
    volatility: float
    beta: float
    concentration_risk: float
    correlation_risk: float

class RiskMonitor:
    """
    Comprehensive real-time risk monitoring system
    """
    
    def __init__(self, max_daily_loss: float = 0.05, max_positions: int = 10,
                 max_concentration: float = 0.3, max_leverage: float = 1.0):
        self.max_daily_loss = max_daily_loss
        self.max_positions = max_positions
        self.max_concentration = max_concentration
        self.max_leverage = max_leverage
        
        # Risk state
        self.positions = {}
        self.daily_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.initial_balance = 100000.0
        self.current_balance = 100000.0
        self.peak_balance = 100000.0
        
        # Risk tracking
        self.risk_alerts = deque(maxlen=1000)
        self.risk_metrics_history = deque(maxlen=1000)
        self.pnl_history = deque(maxlen=1000)
        self.position_history = defaultdict(lambda: deque(maxlen=100))
        
        # Monitoring state
        self.monitoring = False
        self.risk_breaches = {}
        self.emergency_stop = False
        self.last_update = None
        
        # Risk thresholds
        self.risk_thresholds = {
            RiskType.POSITION_SIZE: 0.2,  # 20% max per position
            RiskType.DAILY_LOSS: max_daily_loss,
            RiskType.DRAWDOWN: 0.15,  # 15% max drawdown
            RiskType.CONCENTRATION: max_concentration,
            RiskType.CORRELATION: 0.8,  # Max correlation
            RiskType.VOLATILITY: 0.3,  # 30% annual volatility
            RiskType.LEVERAGE: max_leverage,
            RiskType.LIQUIDITY: 0.1  # 10% of avg volume
        }
        
        logger.info("🛡️  Risk Monitor initialized")
    
    def start_monitoring(self):
        """Start real-time risk monitoring"""
        if self.monitoring:
            logger.warning("⚠️  Risk monitoring already running")
            return
        
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitoring_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        logger.info("🔍 Risk monitoring started")
    
    def stop_monitoring(self):
        """Stop risk monitoring"""
        self.monitoring = False
        logger.info("⏹️  Risk monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Update risk metrics
                self.update_risk_metrics()
                
                # Check all risk thresholds
                self.check_risk_thresholds()
                
                # Handle emergency conditions
                if self.emergency_stop:
                    self.handle_emergency_stop()
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"❌ Risk monitoring error: {e}")
                time.sleep(5)
    
    def update_position(self, symbol: str, amount: float, price: float, 
                       side: str = "long"):
        """Update position information"""
        try:
            current_value = amount * price
            self.positions[symbol] = {
                'amount': amount,
                'price': price,
                'value': current_value,
                'side': side,
                'timestamp': datetime.now()
            }
            
            # Track position history
            self.position_history[symbol].append({
                'timestamp': datetime.now(),
                'amount': amount,
                'price': price,
                'value': current_value
            })
            
            logger.debug(f"📊 Position updated: {symbol} = {amount} @ {price}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update position: {e}")
    
    def update_pnl(self, daily_pnl: float, unrealized_pnl: float):
        """Update P&L information"""
        try:
            self.daily_pnl = daily_pnl
            self.unrealized_pnl = unrealized_pnl
            self.current_balance = self.initial_balance + daily_pnl
            
            # Update peak balance
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
            
            # Track P&L history
            self.pnl_history.append({
                'timestamp': datetime.now(),
                'daily_pnl': daily_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_balance': self.current_balance
            })
            
        except Exception as e:
            logger.error(f"❌ Failed to update P&L: {e}")
    
    def update_risk_metrics(self):
        """Calculate and update comprehensive risk metrics"""
        try:
            # Calculate total exposure
            total_exposure = sum(pos['value'] for pos in self.positions.values())
            
            # Calculate drawdown
            max_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            
            # Calculate volatility (if enough history)
            volatility = 0.0
            if len(self.pnl_history) > 30:
                returns = [h['daily_pnl'] / self.initial_balance 
                          for h in list(self.pnl_history)[-30:]]
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate VaR (95% confidence)
            var_95 = 0.0
            if len(self.pnl_history) > 20:
                pnl_changes = [h['daily_pnl'] for h in list(self.pnl_history)[-20:]]
                var_95 = np.percentile(pnl_changes, 5)  # 5th percentile
            
            # Calculate Sharpe ratio
            sharpe_ratio = 0.0
            if len(self.pnl_history) > 10 and volatility > 0:
                avg_return = np.mean([h['daily_pnl'] / self.initial_balance 
                                    for h in list(self.pnl_history)[-10:]])
                sharpe_ratio = (avg_return * 252) / volatility  # Annualized
            
            # Calculate concentration risk
            concentration_risk = 0.0
            if total_exposure > 0:
                position_weights = [pos['value'] / total_exposure 
                                 for pos in self.positions.values()]
                concentration_risk = max(position_weights) if position_weights else 0
            
            # Calculate correlation risk (simplified)
            correlation_risk = min(len(self.positions) * 0.1, 0.8)
            
            # Create risk metrics
            metrics = RiskMetrics(
                timestamp=datetime.now(),
                total_exposure=total_exposure,
                daily_pnl=self.daily_pnl,
                unrealized_pnl=self.unrealized_pnl,
                max_drawdown=max_drawdown,
                var_95=var_95,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                beta=1.0,  # Simplified
                concentration_risk=concentration_risk,
                correlation_risk=correlation_risk
            )
            
            self.risk_metrics_history.append(metrics)
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Failed to update risk metrics: {e}")
    
    def check_risk_thresholds(self):
        """Check all risk thresholds and generate alerts"""
        try:
            current_metrics = list(self.risk_metrics_history)[-1] if self.risk_metrics_history else None
            if not current_metrics:
                return
            
            # Check daily loss limit
            daily_loss_pct = abs(self.daily_pnl) / self.initial_balance
            if daily_loss_pct > self.risk_thresholds[RiskType.DAILY_LOSS]:
                self.create_risk_alert(
                    RiskType.DAILY_LOSS,
                    RiskLevel.CRITICAL,
                    "PORTFOLIO",
                    daily_loss_pct,
                    self.risk_thresholds[RiskType.DAILY_LOSS],
                    f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.risk_thresholds[RiskType.DAILY_LOSS]:.2%}",
                    "Close all positions and stop trading"
                )
                self.emergency_stop = True
            
            # Check maximum drawdown
            if current_metrics.max_drawdown > self.risk_thresholds[RiskType.DRAWDOWN]:
                self.create_risk_alert(
                    RiskType.DRAWDOWN,
                    RiskLevel.HIGH,
                    "PORTFOLIO",
                    current_metrics.max_drawdown,
                    self.risk_thresholds[RiskType.DRAWDOWN],
                    f"Drawdown {current_metrics.max_drawdown:.2%} exceeds limit",
                    "Reduce position sizes"
                )
            
            # Check concentration risk
            if current_metrics.concentration_risk > self.risk_thresholds[RiskType.CONCENTRATION]:
                self.create_risk_alert(
                    RiskType.CONCENTRATION,
                    RiskLevel.MEDIUM,
                    "PORTFOLIO",
                    current_metrics.concentration_risk,
                    self.risk_thresholds[RiskType.CONCENTRATION],
                    f"Position concentration {current_metrics.concentration_risk:.2%} too high",
                    "Diversify positions"
                )
            
            # Check individual position sizes
            total_value = sum(pos['value'] for pos in self.positions.values())
            for symbol, position in self.positions.items():
                if total_value > 0:
                    position_weight = position['value'] / total_value
                    if position_weight > self.risk_thresholds[RiskType.POSITION_SIZE]:
                        self.create_risk_alert(
                            RiskType.POSITION_SIZE,
                            RiskLevel.MEDIUM,
                            symbol,
                            position_weight,
                            self.risk_thresholds[RiskType.POSITION_SIZE],
                            f"Position {symbol} size {position_weight:.2%} too large",
                            f"Reduce {symbol} position"
                        )
            
            # Check volatility
            if current_metrics.volatility > self.risk_thresholds[RiskType.VOLATILITY]:
                self.create_risk_alert(
                    RiskType.VOLATILITY,
                    RiskLevel.MEDIUM,
                    "PORTFOLIO",
                    current_metrics.volatility,
                    self.risk_thresholds[RiskType.VOLATILITY],
                    f"Portfolio volatility {current_metrics.volatility:.2%} too high",
                    "Reduce position sizes or hedge"
                )
            
        except Exception as e:
            logger.error(f"❌ Risk threshold check failed: {e}")
    
    def create_risk_alert(self, risk_type: RiskType, level: RiskLevel, 
                         symbol: str, current_value: float, threshold: float,
                         message: str, suggested_action: str):
        """Create and log a risk alert"""
        try:
            alert = RiskAlert(
                timestamp=datetime.now(),
                risk_type=risk_type,
                level=level,
                symbol=symbol,
                current_value=current_value,
                threshold=threshold,
                message=message,
                suggested_action=suggested_action
            )
            
            self.risk_alerts.append(alert)
            
            # Log based on severity
            if level == RiskLevel.CRITICAL:
                logger.critical(f"🚨 CRITICAL RISK: {message}")
            elif level == RiskLevel.HIGH:
                logger.error(f"⚠️  HIGH RISK: {message}")
            elif level == RiskLevel.MEDIUM:
                logger.warning(f"⚠️  MEDIUM RISK: {message}")
            else:
                logger.info(f"ℹ️  LOW RISK: {message}")
            
            # Track breach
            self.risk_breaches[risk_type] = alert
            
        except Exception as e:
            logger.error(f"❌ Failed to create risk alert: {e}")
    
    def handle_emergency_stop(self):
        """Handle emergency stop conditions"""
        try:
            logger.critical("🚨 EMERGENCY STOP TRIGGERED")
            
            # Send emergency alerts
            self.create_risk_alert(
                RiskType.DAILY_LOSS,
                RiskLevel.CRITICAL,
                "SYSTEM",
                0,
                0,
                "Emergency stop activated - all trading halted",
                "Manual intervention required"
            )
            
            # Stop monitoring
            self.monitoring = False
            
        except Exception as e:
            logger.error(f"❌ Emergency stop handling failed: {e}")
    
    def reset_emergency_stop(self):
        """Reset emergency stop (manual intervention required)"""
        self.emergency_stop = False
        self.risk_breaches.clear()
        logger.info("✅ Emergency stop reset")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            current_metrics = list(self.risk_metrics_history)[-1] if self.risk_metrics_history else None
            recent_alerts = [alert for alert in list(self.risk_alerts)[-10:]]
            
            return {
                "timestamp": datetime.now().isoformat(),
                "emergency_stop": self.emergency_stop,
                "monitoring_active": self.monitoring,
                "current_metrics": asdict(current_metrics) if current_metrics else None,
                "risk_thresholds": {rt.value: threshold for rt, threshold in self.risk_thresholds.items()},
                "active_breaches": len(self.risk_breaches),
                "recent_alerts": len(recent_alerts),
                "alert_summary": {
                    level.value: len([a for a in recent_alerts if a.level == level])
                    for level in RiskLevel
                },
                "position_count": len(self.positions),
                "total_exposure": sum(pos['value'] for pos in self.positions.values()),
                "daily_pnl": self.daily_pnl,
                "unrealized_pnl": self.unrealized_pnl,
                "account_balance": self.current_balance,
                "last_update": self.last_update.isoformat() if self.last_update else None
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate risk summary: {e}")
            return {"error": str(e)}
    
    def get_recent_alerts(self, count: int = 20) -> List[Dict]:
        """Get recent risk alerts"""
        try:
            recent_alerts = list(self.risk_alerts)[-count:]
            return [asdict(alert) for alert in recent_alerts]
        except Exception as e:
            logger.error(f"❌ Failed to get recent alerts: {e}")
            return []
    
    def update_risk_threshold(self, risk_type: RiskType, new_threshold: float):
        """Update risk threshold"""
        try:
            old_threshold = self.risk_thresholds[risk_type]
            self.risk_thresholds[risk_type] = new_threshold
            logger.info(f"📊 Updated {risk_type.value} threshold: {old_threshold} -> {new_threshold}")
        except Exception as e:
            logger.error(f"❌ Failed to update risk threshold: {e}")
    
    def export_risk_report(self, filepath: str = None) -> str:
        """Export comprehensive risk report"""
        try:
            if not filepath:
                filepath = f"logs/risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "risk_summary": self.get_risk_summary(),
                "risk_metrics_history": [asdict(m) for m in list(self.risk_metrics_history)],
                "risk_alerts": [asdict(a) for a in list(self.risk_alerts)],
                "position_history": {
                    symbol: list(history) 
                    for symbol, history in self.position_history.items()
                },
                "pnl_history": list(self.pnl_history),
                "configuration": {
                    "max_daily_loss": self.max_daily_loss,
                    "max_positions": self.max_positions,
                    "max_concentration": self.max_concentration,
                    "max_leverage": self.max_leverage,
                    "risk_thresholds": {rt.value: threshold for rt, threshold in self.risk_thresholds.items()}
                }
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📋 Risk report exported: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Failed to export risk report: {e}")
            return ""
    
    def calculate_portfolio_var(self, confidence_level: float = 0.95, 
                               time_horizon: int = 1) -> float:
        """Calculate portfolio Value at Risk"""
        try:
            if len(self.pnl_history) < 20:
                return 0.0
            
            # Get recent P&L changes
            pnl_changes = [h['daily_pnl'] for h in list(self.pnl_history)[-60:]]
            
            # Calculate VaR
            percentile = (1 - confidence_level) * 100
            var = np.percentile(pnl_changes, percentile)
            
            # Adjust for time horizon
            var_adjusted = var * np.sqrt(time_horizon)
            
            return abs(var_adjusted)
            
        except Exception as e:
            logger.error(f"❌ VaR calculation failed: {e}")
            return 0.0
    
    def stress_test_portfolio(self, shock_scenarios: Dict[str, float]) -> Dict[str, float]:
        """Run stress test scenarios"""
        try:
            results = {}
            
            for scenario_name, shock_pct in shock_scenarios.items():
                total_loss = 0.0
                
                for symbol, position in self.positions.items():
                    if position['side'] == 'long':
                        position_loss = position['value'] * shock_pct
                    else:  # short position benefits from negative shock
                        position_loss = -position['value'] * shock_pct
                    
                    total_loss += position_loss
                
                results[scenario_name] = {
                    'total_loss': total_loss,
                    'loss_percentage': total_loss / self.current_balance,
                    'remaining_balance': self.current_balance + total_loss
                }
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Stress test failed: {e}")
            return {}

# Factory function for easy initialization
def create_risk_monitor(config: Dict[str, Any]) -> RiskMonitor:
    """Create risk monitor from configuration"""
    return RiskMonitor(
        max_daily_loss=config.get('max_daily_loss', 0.05),
        max_positions=config.get('max_positions', 10),
        max_concentration=config.get('max_concentration', 0.3),
        max_leverage=config.get('max_leverage', 1.0)
    )