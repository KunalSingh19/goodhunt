#!/usr/bin/env python3
"""
Production Logger for GoodHunt v3+
==================================
Comprehensive logging system for production trading operations
"""

import logging
import json
import os
import gzip
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import time
from collections import deque
import psutil

class ProductionLogger:
    """
    Advanced production logging system with rotation, compression, and analytics
    """
    
    def __init__(self, log_dir: str = "logs", max_files: int = 30, 
                 max_size_mb: int = 100, compress_old: bool = True):
        self.log_dir = Path(log_dir)
        self.max_files = max_files
        self.max_size_mb = max_size_mb
        self.compress_old = compress_old
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Log buffers for different categories
        self.trade_logs = deque(maxlen=10000)
        self.performance_logs = deque(maxlen=1000)
        self.error_logs = deque(maxlen=1000)
        self.system_logs = deque(maxlen=1000)
        self.risk_logs = deque(maxlen=1000)
        
        # Setup logging
        self._setup_loggers()
        
        # Start background tasks
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_logs)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
        
        self.log_info("🗂️  Production Logger initialized")
    
    def _setup_loggers(self):
        """Setup specialized loggers for different categories"""
        
        # Main system logger
        self.logger = logging.getLogger('GoodHunt.Production')
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(lineno)-4d | %(message)s'
        )
        
        # Daily rotating file handler
        today = datetime.now().strftime('%Y%m%d')
        
        # Main log file
        main_handler = logging.FileHandler(self.log_dir / f"goodhunt_{today}.log")
        main_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(main_handler)
        
        # Specialized loggers
        self._setup_specialized_logger('trades', 'trade')
        self._setup_specialized_logger('performance', 'performance')
        self._setup_specialized_logger('errors', 'error')
        self._setup_specialized_logger('system', 'system')
        self._setup_specialized_logger('risk', 'risk')
    
    def _setup_specialized_logger(self, category: str, name: str):
        """Setup specialized logger for specific category"""
        logger = logging.getLogger(f'GoodHunt.{category.title()}')
        logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s'
        )
        
        today = datetime.now().strftime('%Y%m%d')
        handler = logging.FileHandler(self.log_dir / f"{name}_{today}.log")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        setattr(self, f"{name}_logger", logger)
    
    # Trade logging methods
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade execution"""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'type': 'TRADE',
                'data': trade_data
            }
            
            self.trade_logs.append(log_entry)
            
            # Log to trade file
            self.trade_logger.info(json.dumps(log_entry, default=str))
            
            # Log summary to main log
            symbol = trade_data.get('symbol', 'UNKNOWN')
            action = trade_data.get('action', 'UNKNOWN')
            pnl = trade_data.get('pnl', 0)
            self.log_info(f"💰 TRADE: {action} {symbol} | P&L: ${pnl:.2f}")
            
        except Exception as e:
            self.log_error(f"Failed to log trade: {e}")
    
    def log_order(self, order_data: Dict[str, Any]):
        """Log order placement/execution"""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'type': 'ORDER',
                'data': order_data
            }
            
            self.trade_logs.append(log_entry)
            self.trade_logger.info(json.dumps(log_entry, default=str))
            
        except Exception as e:
            self.log_error(f"Failed to log order: {e}")
    
    # Performance logging
    def log_performance(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'type': 'PERFORMANCE',
                'data': metrics
            }
            
            self.performance_logs.append(log_entry)
            self.performance_logger.info(json.dumps(log_entry, default=str))
            
        except Exception as e:
            self.log_error(f"Failed to log performance: {e}")
    
    def log_portfolio_update(self, portfolio_data: Dict[str, Any]):
        """Log portfolio state changes"""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'type': 'PORTFOLIO',
                'data': portfolio_data
            }
            
            self.performance_logs.append(log_entry)
            self.performance_logger.info(json.dumps(log_entry, default=str))
            
        except Exception as e:
            self.log_error(f"Failed to log portfolio: {e}")
    
    # Risk logging
    def log_risk_alert(self, alert_data: Dict[str, Any]):
        """Log risk alerts"""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'type': 'RISK_ALERT',
                'data': alert_data
            }
            
            self.risk_logs.append(log_entry)
            self.risk_logger.warning(json.dumps(log_entry, default=str))
            
            # Log to main log based on severity
            level = alert_data.get('level', 'medium')
            message = alert_data.get('message', 'Risk alert')
            
            if level == 'critical':
                self.log_critical(f"🚨 RISK ALERT: {message}")
            elif level == 'high':
                self.log_error(f"⚠️  RISK ALERT: {message}")
            else:
                self.log_warning(f"⚠️  RISK ALERT: {message}")
                
        except Exception as e:
            self.log_error(f"Failed to log risk alert: {e}")
    
    def log_risk_metrics(self, metrics: Dict[str, Any]):
        """Log risk metrics"""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'type': 'RISK_METRICS',
                'data': metrics
            }
            
            self.risk_logs.append(log_entry)
            self.risk_logger.info(json.dumps(log_entry, default=str))
            
        except Exception as e:
            self.log_error(f"Failed to log risk metrics: {e}")
    
    # System logging
    def log_system_event(self, event_data: Dict[str, Any]):
        """Log system events"""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'type': 'SYSTEM_EVENT',
                'data': event_data
            }
            
            self.system_logs.append(log_entry)
            self.system_logger.info(json.dumps(log_entry, default=str))
            
        except Exception as e:
            self.log_error(f"Failed to log system event: {e}")
    
    def log_system_metrics(self):
        """Log system performance metrics"""
        try:
            metrics = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'network_io': dict(psutil.net_io_counters()._asdict()),
                'process_count': len(psutil.pids()),
                'timestamp': datetime.now().isoformat()
            }
            
            self.log_system_event({
                'category': 'system_metrics',
                'metrics': metrics
            })
            
        except Exception as e:
            self.log_error(f"Failed to log system metrics: {e}")
    
    # Standard logging methods
    def log_info(self, message: str, extra_data: Dict = None):
        """Log info message"""
        if extra_data:
            message = f"{message} | Data: {json.dumps(extra_data, default=str)}"
        self.logger.info(message)
    
    def log_warning(self, message: str, extra_data: Dict = None):
        """Log warning message"""
        if extra_data:
            message = f"{message} | Data: {json.dumps(extra_data, default=str)}"
        self.logger.warning(message)
    
    def log_error(self, message: str, extra_data: Dict = None):
        """Log error message"""
        if extra_data:
            message = f"{message} | Data: {json.dumps(extra_data, default=str)}"
        
        timestamp = datetime.now().isoformat()
        error_entry = {
            'timestamp': timestamp,
            'type': 'ERROR',
            'message': message,
            'data': extra_data or {}
        }
        
        self.error_logs.append(error_entry)
        self.error_logger.error(json.dumps(error_entry, default=str))
        self.logger.error(message)
    
    def log_critical(self, message: str, extra_data: Dict = None):
        """Log critical message"""
        if extra_data:
            message = f"{message} | Data: {json.dumps(extra_data, default=str)}"
        
        timestamp = datetime.now().isoformat()
        error_entry = {
            'timestamp': timestamp,
            'type': 'CRITICAL',
            'message': message,
            'data': extra_data or {}
        }
        
        self.error_logs.append(error_entry)
        self.error_logger.critical(json.dumps(error_entry, default=str))
        self.logger.critical(message)
    
    # Analytics and reporting
    def generate_daily_report(self, date: str = None) -> str:
        """Generate daily activity report"""
        try:
            if not date:
                date = datetime.now().strftime('%Y%m%d')
            
            # Collect data from logs
            trades_today = [log for log in self.trade_logs 
                          if log['timestamp'].startswith(date[:4] + '-' + date[4:6] + '-' + date[6:])]
            
            errors_today = [log for log in self.error_logs 
                          if log['timestamp'].startswith(date[:4] + '-' + date[4:6] + '-' + date[6:])]
            
            # Generate report
            report = {
                'date': date,
                'generated_at': datetime.now().isoformat(),
                'summary': {
                    'total_trades': len(trades_today),
                    'total_errors': len(errors_today),
                    'system_uptime_hours': 24,  # Placeholder
                    'memory_usage_avg': 50.0,  # Placeholder
                    'cpu_usage_avg': 25.0      # Placeholder
                },
                'trades': trades_today[-10:],  # Last 10 trades
                'errors': errors_today,
                'performance_metrics': list(self.performance_logs)[-10:]
            }
            
            # Save report
            report_file = self.log_dir / f"daily_report_{date}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.log_info(f"📋 Daily report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.log_error(f"Failed to generate daily report: {e}")
            return ""
    
    def get_log_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get log summary for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cutoff_str = cutoff_time.isoformat()
            
            # Filter logs by time
            recent_trades = [log for log in self.trade_logs if log['timestamp'] >= cutoff_str]
            recent_errors = [log for log in self.error_logs if log['timestamp'] >= cutoff_str]
            recent_risks = [log for log in self.risk_logs if log['timestamp'] >= cutoff_str]
            
            return {
                'period_hours': hours,
                'trades_count': len(recent_trades),
                'errors_count': len(recent_errors),
                'risk_alerts_count': len(recent_risks),
                'trade_pnl_total': sum(log['data'].get('pnl', 0) for log in recent_trades 
                                     if log['type'] == 'TRADE'),
                'error_types': list(set(log['data'].get('type', 'unknown') for log in recent_errors)),
                'most_traded_symbols': self._get_most_frequent([
                    log['data'].get('symbol') for log in recent_trades 
                    if log['type'] == 'TRADE' and log['data'].get('symbol')
                ], 5),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.log_error(f"Failed to get log summary: {e}")
            return {'error': str(e)}
    
    def _get_most_frequent(self, items: List[str], top_n: int = 5) -> List[Dict]:
        """Get most frequent items from list"""
        from collections import Counter
        counter = Counter(items)
        return [{'item': item, 'count': count} for item, count in counter.most_common(top_n)]
    
    def _cleanup_old_logs(self):
        """Background task to cleanup old log files"""
        while self.running:
            try:
                # Sleep for 1 hour between cleanups
                time.sleep(3600)
                
                # Get all log files
                log_files = list(self.log_dir.glob("*.log"))
                
                # Sort by modification time
                log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                
                # Remove old files beyond max_files limit
                if len(log_files) > self.max_files:
                    files_to_remove = log_files[self.max_files:]
                    for file in files_to_remove:
                        if self.compress_old and not file.name.endswith('.gz'):
                            # Compress before removing
                            self._compress_file(file)
                        else:
                            file.unlink()
                            self.log_info(f"🗑️  Removed old log file: {file.name}")
                
                # Check file sizes and rotate if needed
                for file in log_files[:self.max_files]:
                    if file.stat().st_size > self.max_size_mb * 1024 * 1024:
                        self._rotate_file(file)
                
            except Exception as e:
                self.log_error(f"Log cleanup error: {e}")
    
    def _compress_file(self, file_path: Path):
        """Compress log file"""
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            file_path.unlink()
            self.log_info(f"🗜️  Compressed log file: {file_path.name} -> {compressed_path.name}")
            
        except Exception as e:
            self.log_error(f"Failed to compress file {file_path}: {e}")
    
    def _rotate_file(self, file_path: Path):
        """Rotate large log file"""
        try:
            timestamp = datetime.now().strftime('%H%M%S')
            rotated_path = file_path.with_stem(f"{file_path.stem}_{timestamp}")
            file_path.rename(rotated_path)
            
            if self.compress_old:
                self._compress_file(rotated_path)
            
            self.log_info(f"🔄 Rotated log file: {file_path.name}")
            
        except Exception as e:
            self.log_error(f"Failed to rotate file {file_path}: {e}")
    
    def shutdown(self):
        """Shutdown the production logger"""
        self.running = False
        self.log_info("🔄 Production Logger shutting down")
        
        # Generate final report
        self.generate_daily_report()
        
        # Close all handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

# Global production logger instance
production_logger = None

def get_production_logger() -> ProductionLogger:
    """Get global production logger instance"""
    global production_logger
    if production_logger is None:
        production_logger = ProductionLogger()
    return production_logger

def log_trade(trade_data: Dict[str, Any]):
    """Convenience function to log trade"""
    get_production_logger().log_trade(trade_data)

def log_performance(metrics: Dict[str, Any]):
    """Convenience function to log performance"""
    get_production_logger().log_performance(metrics)

def log_risk_alert(alert_data: Dict[str, Any]):
    """Convenience function to log risk alert"""
    get_production_logger().log_risk_alert(alert_data)

def log_error(message: str, extra_data: Dict = None):
    """Convenience function to log error"""
    get_production_logger().log_error(message, extra_data)

def log_info(message: str, extra_data: Dict = None):
    """Convenience function to log info"""
    get_production_logger().log_info(message, extra_data)