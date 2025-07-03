#!/usr/bin/env python3
"""
GoodHunt v3+ Production System Test
===================================
Test script to demonstrate the production system functionality
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Simulate psutil for testing
class MockPsutil:
    @staticmethod
    def cpu_percent(interval=1):
        return 25.5
    
    @staticmethod
    def virtual_memory():
        class Memory:
            percent = 45.2
        return Memory()
    
    @staticmethod
    def disk_usage(path):
        class Disk:
            percent = 67.8
        return Disk()
    
    @staticmethod
    def net_io_counters():
        class NetIO:
            bytes_sent = 1024000
            bytes_recv = 2048000
            packets_sent = 5000
            packets_recv = 6000
            
            def _asdict(self):
                return {
                    'bytes_sent': self.bytes_sent,
                    'bytes_recv': self.bytes_recv,
                    'packets_sent': self.packets_sent,
                    'packets_recv': self.packets_recv
                }
        return NetIO()
    
    @staticmethod
    def pids():
        return list(range(100, 200))  # Simulate 100 processes

# Mock the psutil module
sys.modules['psutil'] = MockPsutil()

# Now import our production system
from logs.production_logger import ProductionLogger
from utils.config import ConfigManager
from live.broker_connector import PaperTradingBroker
from live.risk_monitor import RiskMonitor
from live.performance_tracker import PerformanceTracker
from live.alert_manager import AlertManager

def test_production_logger():
    """Test the production logger"""
    print("🧪 Testing Production Logger...")
    
    logger = ProductionLogger(log_dir="test_logs")
    
    # Test trade logging
    logger.log_trade({
        'symbol': 'AAPL',
        'action': 'BUY',
        'quantity': 100,
        'price': 150.0,
        'pnl': 50.0,
        'commission': 1.0
    })
    
    # Test performance logging
    logger.log_performance({
        'total_return': 15.5,
        'sharpe_ratio': 1.8,
        'max_drawdown': 5.2
    })
    
    # Test risk alert
    logger.log_risk_alert({
        'level': 'medium',
        'message': 'Position concentration exceeds 20%',
        'symbol': 'AAPL'
    })
    
    # Generate summary
    summary = logger.get_log_summary(hours=1)
    print(f"✅ Logger Test Complete - Summary: {summary['trades_count']} trades logged")
    
    return True

def test_config_manager():
    """Test the configuration manager"""
    print("🧪 Testing Configuration Manager...")
    
    config_manager = ConfigManager(config_path="test_config.yaml")
    
    # Test default config creation
    config = config_manager.get_default_config()
    
    # Test saving and loading
    config_manager.save_config(config)
    loaded_config = config_manager.load_config()
    
    # Test feature access
    enabled_features = config_manager.get_enabled_features()
    
    print(f"✅ Config Test Complete - {len(enabled_features)} feature categories loaded")
    
    return True

def test_broker_connector():
    """Test the broker connector with paper trading"""
    print("🧪 Testing Broker Connector (Paper Trading)...")
    
    import asyncio
    
    async def run_broker_test():
        broker = PaperTradingBroker(initial_balance=100000.0)
        
        # Connect
        await broker.connect()
        
        # Place test order
        from live.broker_connector import OrderSide, OrderType
        order = await broker.place_order("AAPL", OrderSide.BUY, 100, OrderType.MARKET, 150.0)
        
        # Check balance
        balance = await broker.get_balance()
        
        # Get positions
        positions = await broker.get_positions()
        
        print(f"✅ Broker Test Complete - Order ID: {order.id}, Balance: ${balance['USD']:.2f}")
        
        return True
    
    return asyncio.run(run_broker_test())

def test_risk_monitor():
    """Test the risk monitoring system"""
    print("🧪 Testing Risk Monitor...")
    
    risk_monitor = RiskMonitor(
        max_daily_loss=0.05,
        max_positions=10,
        max_concentration=0.3
    )
    
    # Update position
    risk_monitor.update_position("AAPL", 100, 150.0, "long")
    
    # Update P&L
    risk_monitor.update_pnl(daily_pnl=500.0, unrealized_pnl=200.0)
    
    # Get risk summary
    summary = risk_monitor.get_risk_summary()
    
    print(f"✅ Risk Monitor Test Complete - Status: {summary['emergency_stop']}")
    
    return True

def test_performance_tracker():
    """Test the performance tracking system"""
    print("🧪 Testing Performance Tracker...")
    
    tracker = PerformanceTracker(initial_balance=100000.0)
    
    # Add test trades
    for i in range(5):
        tracker.update_trade({
            'symbol': 'AAPL',
            'side': 'BUY' if i % 2 == 0 else 'SELL',
            'quantity': 100,
            'price': 150.0 + i,
            'pnl': (i - 2) * 100,  # Mix of profits and losses
            'commission': 1.0,
            'net_pnl': (i - 2) * 100 - 1.0
        })
    
    # Get performance summary
    summary = tracker.get_performance_summary()
    
    print(f"✅ Performance Tracker Test Complete - Total Return: {summary['total_return_pct']:.2f}%")
    
    return True

def test_alert_manager():
    """Test the alert management system"""
    print("🧪 Testing Alert Manager...")
    
    alert_manager = AlertManager()
    
    # Configure with mock settings
    alert_manager.configure({
        'email': {'enabled': False},
        'slack': {'enabled': False},
        'discord': {'enabled': False}
    })
    
    # Send test alerts
    alert_manager.info("System Started", "GoodHunt production system initialized")
    alert_manager.warning("High Volume", "Trading volume above average", symbol="AAPL")
    alert_manager.error("Connection Issue", "Temporary connection problem", symbol="BTC-USD")
    
    # Get recent alerts
    recent_alerts = alert_manager.get_recent_alerts(count=5)
    
    print(f"✅ Alert Manager Test Complete - {len(recent_alerts)} alerts generated")
    
    return True

def test_feature_matrix():
    """Test the feature matrix and configuration"""
    print("🧪 Testing Feature Matrix...")
    
    try:
        import pandas as pd
        feature_df = pd.read_csv('FEATURE_MATRIX.csv')
        
        # Count features by category
        category_counts = feature_df['Category'].value_counts()
        total_features = len(feature_df)
        
        print(f"✅ Feature Matrix Test Complete - {total_features} features across {len(category_counts)} categories")
        
        # Show category breakdown
        for category, count in category_counts.head().items():
            print(f"   {category}: {count} features")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Feature Matrix Test: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test of the production system"""
    
    print("🚀 GoodHunt v3+ Production System Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Production Logger", test_production_logger),
        ("Configuration Manager", test_config_manager),
        ("Broker Connector", test_broker_connector),
        ("Risk Monitor", test_risk_monitor),
        ("Performance Tracker", test_performance_tracker),
        ("Alert Manager", test_alert_manager),
        ("Feature Matrix", test_feature_matrix)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
            print()
        except Exception as e:
            print(f"❌ {test_name} Test Failed: {e}")
            test_results.append((test_name, False))
            print()
    
    # Print summary
    print("📊 Test Results Summary")
    print("-" * 30)
    
    passed = sum(1 for name, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print("-" * 30)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Production system is ready.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    # Generate test report
    report = {
        'test_timestamp': datetime.now().isoformat(),
        'total_tests': total,
        'passed_tests': passed,
        'success_rate': passed/total*100,
        'test_results': [
            {'test': name, 'passed': result} for name, result in test_results
        ],
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Save test report
    os.makedirs('test_results', exist_ok=True)
    report_file = f"test_results/production_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📋 Test report saved: {report_file}")
    
    return passed == total

if __name__ == "__main__":
    # Ensure we have required directories
    for directory in ['logs', 'test_logs', 'test_results']:
        os.makedirs(directory, exist_ok=True)
    
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)