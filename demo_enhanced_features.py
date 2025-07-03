#!/usr/bin/env python3
"""
🚀 GoodHunt v3+ Enhanced Features Demo
Demonstrates all 20 high-impact profitability-enhancing features

Features implemented:
🧠 OBSERVATION ENHANCEMENTS (1-3)
⚙️ ENVIRONMENT SIGNAL CONTROLS (4-6)  
🧮 ADVANCED REWARD ADDITIONS (7-10)
🎯 TRADE EXECUTION CONTROLS (11-14)
📊 ANALYTICS & MONITORING (15-18)
🧠 STRATEGY ADAPTATION (19-20)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta

from env.trading_env import TradingEnv
from logs.analyze import TradingAnalytics

def create_sample_data(symbol="AAPL", period="1y"):
    """Create sample trading data for demonstration"""
    print(f"📥 Downloading sample data for {symbol}...")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        print(f"✅ Downloaded {len(df)} data points")
        return df.reset_index()
        
    except Exception as e:
        print(f"⚠️ Error downloading data: {e}")
        print("📊 Creating synthetic data instead...")
        
        # Create synthetic data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        
        price = 100
        prices = []
        volumes = []
        
        for i in range(len(dates)):
            price *= (1 + np.random.normal(0.001, 0.02))
            prices.append(price)
            volumes.append(np.random.randint(1000000, 10000000))
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': [p * np.random.uniform(0.995, 1.005) for p in prices],
            'High': [p * np.random.uniform(1.001, 1.02) for p in prices],
            'Low': [p * np.random.uniform(0.98, 0.999) for p in prices],
            'Close': prices,
            'Volume': volumes
        })
        
        return df

def demonstrate_observation_enhancements(df):
    """Demonstrate Features 1-3: Observation Enhancements"""
    print("\n🧠 OBSERVATION ENHANCEMENTS")
    print("=" * 50)
    
    # Import and apply indicators
    from utils.indicators import add_all_indicators
    df_enhanced = add_all_indicators(df.copy())
    
    # Feature 1: VWAP
    if 'vwap' in df_enhanced.columns:
        latest_vwap = df_enhanced['vwap'].iloc[-1]
        latest_price = df_enhanced['Close'].iloc[-1]
        print(f"📈 Feature 1 - VWAP: ${latest_vwap:.2f} vs Price: ${latest_price:.2f}")
        print(f"   Price vs VWAP: {((latest_price/latest_vwap-1)*100):+.2f}%")
    
    # Feature 2: Beta 
    if 'beta' in df_enhanced.columns:
        latest_beta = df_enhanced['beta'].iloc[-1]
        print(f"📊 Feature 2 - Beta: {latest_beta:.2f}")
        risk_level = "HIGH" if abs(latest_beta) > 1.5 else "MEDIUM" if abs(latest_beta) > 1.0 else "LOW"
        print(f"   Risk Level: {risk_level}")
    
    # Feature 3: Fisher Transform RSI
    if 'rsi_fisher' in df_enhanced.columns:
        latest_fisher = df_enhanced['rsi_fisher'].iloc[-1]
        print(f"🎣 Feature 3 - Fisher RSI: {latest_fisher:.2f}")
        signal = "OVERSOLD" if latest_fisher < -2 else "OVERBOUGHT" if latest_fisher > 2 else "NEUTRAL"
        print(f"   Signal: {signal}")
    
    return df_enhanced

def demonstrate_trading_simulation(df):
    """Demonstrate Features 4-20: Trading Environment Features"""
    print("\n⚙️ TRADING SIMULATION WITH ENHANCED FEATURES")
    print("=" * 50)
    
    # Create enhanced trading environment
    config = {
        'max_drawdown': 0.15,
        'enable_multi_asset': True,
        'volatility_regime_switching': True
    }
    
    env = TradingEnv(
        df=df,
        window_size=30,
        initial_balance=10000.0,
        max_exposure=0.8,
        fee_pct=0.001,
        slippage_model="dynamic",
        config=config
    )
    
    # Run simulation
    print("🎮 Running trading simulation...")
    obs, _ = env.reset()
    
    total_reward = 0
    actions_taken = []
    
    for step in range(min(100, len(df) - env.window_size - 1)):
        # Simple strategy: buy when RSI < 30, sell when RSI > 70
        current_data = df.iloc[env.current_step]
        
        if 'rsi' in df.columns:
            rsi = current_data.get('rsi', 50)
            if rsi < 30 and env.position_size == 0:
                action = 1  # BUY
            elif rsi > 70 and env.position_size > 0:
                action = 2  # SELL
            elif env.position_size > 0 and np.random.random() < 0.1:
                action = 3  # SCALE_UP occasionally
            else:
                action = 0  # HOLD
        else:
            action = np.random.choice([0, 1, 2])  # Random action if no RSI
        
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        actions_taken.append(env.ACTIONS[action])
        
        if done:
            break
    
    # Print results
    print(f"✅ Simulation completed!")
    print(f"   Initial Balance: ${env.initial_balance:.2f}")
    print(f"   Final Net Worth: ${env.net_worth:.2f}")
    print(f"   Total Return: {((env.net_worth/env.initial_balance-1)*100):+.2f}%")
    print(f"   Total Trades: {len(env.trades)}")
    print(f"   Max Drawdown: {(env.drawdown*100):.2f}%")
    
    # Save results for analytics
    Path("backtest").mkdir(exist_ok=True)
    env.save_trades()
    env.save_equity_curve()
    
    return env

def demonstrate_analytics_features():
    """Demonstrate Features 15-18: Analytics & Monitoring"""
    print("\n📊 ANALYTICS & MONITORING")
    print("=" * 50)
    
    # Create analytics instance
    analytics = TradingAnalytics()
    
    # Generate summary statistics
    analytics.print_summary_stats()
    
    # Generate all analytics charts
    print("\n🎨 Generating visualization reports...")
    try:
        analytics.generate_full_report()
    except Exception as e:
        print(f"⚠️ Note: Some visualizations may not display in headless mode: {e}")
        print("📁 Charts will be saved to logs/ directory")

def print_feature_summary():
    """Print summary of all implemented features"""
    print("\n🎯 IMPLEMENTED FEATURES SUMMARY")
    print("=" * 50)
    
    features = {
        "🧠 OBSERVATION ENHANCEMENTS": [
            "1. Volume-Weighted Average Price (VWAP)",
            "2. Beta vs Benchmark", 
            "3. Fisher Transform of RSI"
        ],
        "⚙️ ENVIRONMENT SIGNAL CONTROLS": [
            "4. Multi-Asset Exposure Control",
            "5. Anti-Chasing Logic",
            "6. Spread Simulation (Bid-Ask)"
        ],
        "🧮 ADVANCED REWARD ADDITIONS": [
            "7. Reward Decay on Time",
            "8. Slippage Penalty", 
            "9. Stop-Loss Breach Penalty",
            "10. Profit Streak Bonus"
        ],
        "🎯 TRADE EXECUTION CONTROLS": [
            "11. Fractional Trading",
            "12. Partial Position Scaling",
            "13. Multi-Step Reward Averaging",
            "14. End-Of-Day Forced Close"
        ],
        "📊 ANALYTICS & MONITORING": [
            "15. Trading Volume Heatmap",
            "16. Trade Duration Distribution", 
            "17. Risk Contribution Waterfall",
            "18. Drawdown Chart"
        ],
        "🧠 STRATEGY ADAPTATION": [
            "19. Volatility Regime Switcher",
            "20. On-the-Fly Hyperparameter Adjuster"
        ]
    }
    
    for category, feature_list in features.items():
        print(f"\n{category}")
        for feature in feature_list:
            print(f"  ✅ {feature}")

def main():
    """Main demonstration function"""
    print("🚀 GoodHunt v3+ Enhanced Features Demonstration")
    print("=" * 60)
    print("Demonstrating 20 high-impact profitability-enhancing features")
    print("=" * 60)
    
    # Step 1: Create sample data
    df = create_sample_data("AAPL", "6mo")
    
    # Step 2: Demonstrate observation enhancements
    df_enhanced = demonstrate_observation_enhancements(df)
    
    # Step 3: Run trading simulation
    env = demonstrate_trading_simulation(df_enhanced)
    
    # Step 4: Generate analytics
    demonstrate_analytics_features()
    
    # Step 5: Print feature summary
    print_feature_summary()
    
    print("\n🎉 DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("📁 Check the following directories for outputs:")
    print("   • backtest/ - Trading results")
    print("   • logs/ - Analytics charts and reports") 
    print("\n💡 TIP: Run this demo multiple times with different")
    print("   market conditions to see feature adaptation!")


if __name__ == "__main__":
    main()