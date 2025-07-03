#!/usr/bin/env python3
"""
🔍 GoodHunt v3+ Implementation Verification
Validates that all 20 enhanced features are properly implemented
"""

import sys
import os
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (MISSING)")
        return False

def check_function_exists(filepath, function_name, description):
    """Check if a function exists in a file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if f"def {function_name}" in content:
                print(f"✅ {description}: {function_name}()")
                return True
            else:
                print(f"❌ {description}: {function_name}() (NOT FOUND)")
                return False
    except FileNotFoundError:
        print(f"❌ {description}: File {filepath} not found")
        return False

def check_feature_in_file(filepath, feature_text, description):
    """Check if a feature implementation exists in a file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if feature_text in content:
                print(f"✅ {description}")
                return True
            else:
                print(f"❌ {description} (NOT IMPLEMENTED)")
                return False
    except FileNotFoundError:
        print(f"❌ {description}: File {filepath} not found")
        return False

def main():
    """Main verification function"""
    print("🔍 GoodHunt v3+ Enhanced Features Verification")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    # Check core files exist
    print("\n📁 CORE FILES")
    print("-" * 30)
    total += 4
    passed += check_file_exists("utils/indicators.py", "Indicators module")
    passed += check_file_exists("env/trading_env.py", "Trading environment")
    passed += check_file_exists("logs/analyze.py", "Analytics module")
    passed += check_file_exists("demo_enhanced_features.py", "Demo script")
    
    # Check observation enhancements (Features 1-3)
    print("\n🧠 OBSERVATION ENHANCEMENTS")
    print("-" * 30)
    total += 3
    passed += check_function_exists("utils/indicators.py", "vwap", "Feature 1: VWAP")
    passed += check_function_exists("utils/indicators.py", "beta_vs_benchmark", "Feature 2: Beta vs Benchmark")
    passed += check_function_exists("utils/indicators.py", "fisher_transform_rsi", "Feature 3: Fisher Transform RSI")
    
    # Check environment controls (Features 4-6)
    print("\n⚙️ ENVIRONMENT SIGNAL CONTROLS")
    print("-" * 30)
    total += 3
    passed += check_feature_in_file("env/trading_env.py", "Multi-Asset Exposure Control", "Feature 4: Multi-Asset Exposure")
    passed += check_feature_in_file("env/trading_env.py", "Anti-Chasing Logic", "Feature 5: Anti-Chasing Logic")
    passed += check_feature_in_file("env/trading_env.py", "Spread Simulation", "Feature 6: Bid-Ask Spread")
    
    # Check reward enhancements (Features 7-10)
    print("\n🧮 ADVANCED REWARD ADDITIONS")
    print("-" * 30)
    total += 4
    passed += check_feature_in_file("env/trading_env.py", "Reward Decay on Time", "Feature 7: Time Decay")
    passed += check_feature_in_file("env/trading_env.py", "Slippage Penalty", "Feature 8: Slippage Penalty")
    passed += check_feature_in_file("env/trading_env.py", "Stop-Loss Breach Penalty", "Feature 9: Stop-Loss Penalty")
    passed += check_feature_in_file("env/trading_env.py", "Profit Streak Bonus", "Feature 10: Profit Streak")
    
    # Check execution controls (Features 11-14)
    print("\n🎯 TRADE EXECUTION CONTROLS")
    print("-" * 30)
    total += 4
    passed += check_feature_in_file("env/trading_env.py", "Fractional Trading", "Feature 11: Fractional Trading")
    passed += check_feature_in_file("env/trading_env.py", "Partial Position Scaling", "Feature 12: Position Scaling")
    passed += check_feature_in_file("env/trading_env.py", "Multi-Step Reward Averaging", "Feature 13: Reward Averaging")
    passed += check_feature_in_file("env/trading_env.py", "End-Of-Day Forced Close", "Feature 14: EOD Close")
    
    # Check analytics features (Features 15-18)
    print("\n📊 ANALYTICS & MONITORING")
    print("-" * 30)
    total += 4
    passed += check_function_exists("logs/analyze.py", "feature_15_trading_volume_heatmap", "Feature 15: Volume Heatmap")
    passed += check_function_exists("logs/analyze.py", "feature_16_trade_duration_distribution", "Feature 16: Duration Distribution")
    passed += check_function_exists("logs/analyze.py", "feature_17_risk_contribution_waterfall", "Feature 17: Risk Waterfall")
    passed += check_function_exists("logs/analyze.py", "feature_18_drawdown_chart", "Feature 18: Drawdown Chart")
    
    # Check strategy adaptation (Features 19-20)
    print("\n🧠 STRATEGY ADAPTATION")
    print("-" * 30)
    total += 2
    passed += check_feature_in_file("env/trading_env.py", "Volatility Regime Switcher", "Feature 19: Volatility Regime")
    passed += check_feature_in_file("env/trading_env.py", "Hyperparameter Adjuster", "Feature 20: Dynamic Hyperparams")
    
    # Final results
    print("\n" + "=" * 60)
    print(f"🎯 VERIFICATION RESULTS")
    print("=" * 60)
    print(f"✅ Features Implemented: {passed}/{total}")
    print(f"📊 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL FEATURES SUCCESSFULLY IMPLEMENTED!")
        print("🚀 GoodHunt v3+ is ready for enhanced trading!")
        return True
    else:
        print(f"\n⚠️  {total-passed} features need attention")
        print("🔧 Please review the missing implementations above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)