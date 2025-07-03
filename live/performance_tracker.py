#!/usr/bin/env python3
"""
Performance Tracker for GoodHunt v3+
====================================
Real-time performance monitoring and analytics
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from collections import deque

logger = logging.getLogger('GoodHunt.PerformanceTracker')

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    total_return: float
    daily_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int

class PerformanceTracker:
    """Real-time performance tracking and analytics"""
    
    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        
        # Trade tracking
        self.trades = deque(maxlen=10000)
        self.daily_returns = deque(maxlen=252)  # One year
        self.equity_curve = deque(maxlen=10000)
        
        # Performance metrics
        self.metrics_history = deque(maxlen=1000)
        self.last_update = None
        
        # Tracking state
        self.tracking = False
        
        logger.info("📊 Performance Tracker initialized")
    
    def update_trade(self, trade_data: Dict[str, Any]):
        """Update with new trade data"""
        try:
            self.trades.append({
                'timestamp': datetime.now(),
                'symbol': trade_data.get('symbol', ''),
                'side': trade_data.get('side', ''),
                'quantity': trade_data.get('quantity', 0),
                'price': trade_data.get('price', 0),
                'pnl': trade_data.get('pnl', 0),
                'commission': trade_data.get('commission', 0),
                'net_pnl': trade_data.get('net_pnl', 0)
            })
            
            # Update balance
            self.current_balance += trade_data.get('net_pnl', 0)
            self.equity_curve.append({
                'timestamp': datetime.now(),
                'balance': self.current_balance
            })
            
            # Update peak balance
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
            
            # Calculate and store metrics
            self.calculate_metrics()
            
        except Exception as e:
            logger.error(f"❌ Failed to update trade: {e}")
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            if not self.trades:
                return None
            
            trades_list = list(self.trades)
            
            # Basic metrics
            total_return = (self.current_balance - self.initial_balance) / self.initial_balance
            daily_return = 0.0
            if len(self.equity_curve) > 1:
                yesterday_balance = list(self.equity_curve)[-2]['balance']
                daily_return = (self.current_balance - yesterday_balance) / yesterday_balance
            
            # Trade statistics
            winning_trades = [t for t in trades_list if t['net_pnl'] > 0]
            losing_trades = [t for t in trades_list if t['net_pnl'] < 0]
            
            total_trades = len(trades_list)
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            win_rate = win_count / total_trades if total_trades > 0 else 0
            
            # P&L statistics
            avg_win = np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['net_pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win * win_count / (avg_loss * loss_count)) if avg_loss != 0 and loss_count > 0 else 0
            
            # Risk metrics
            max_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            
            # Sharpe ratio
            sharpe_ratio = 0.0
            if len(self.daily_returns) > 30:
                returns = np.array(list(self.daily_returns))
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Sortino ratio
            sortino_ratio = 0.0
            if len(self.daily_returns) > 30:
                returns = np.array(list(self.daily_returns))
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    downside_deviation = np.std(negative_returns)
                    sortino_ratio = np.mean(returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                total_return=total_return,
                daily_return=daily_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                total_trades=total_trades,
                winning_trades=win_count,
                losing_trades=loss_count
            )
            
            self.metrics_history.append(metrics)
            self.last_update = datetime.now()
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate metrics: {e}")
            return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            current_metrics = list(self.metrics_history)[-1] if self.metrics_history else None
            
            return {
                "timestamp": datetime.now().isoformat(),
                "current_balance": self.current_balance,
                "initial_balance": self.initial_balance,
                "peak_balance": self.peak_balance,
                "total_return_pct": ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
                "current_metrics": asdict(current_metrics) if current_metrics else None,
                "trade_count": len(self.trades),
                "last_update": self.last_update.isoformat() if self.last_update else None
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate performance summary: {e}")
            return {"error": str(e)}
    
    def export_performance_report(self, filepath: str = None) -> str:
        """Export detailed performance report"""
        try:
            if not filepath:
                filepath = f"logs/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "performance_summary": self.get_performance_summary(),
                "metrics_history": [asdict(m) for m in list(self.metrics_history)],
                "trades": list(self.trades),
                "equity_curve": list(self.equity_curve),
                "daily_returns": list(self.daily_returns)
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📋 Performance report exported: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Failed to export performance report: {e}")
            return ""