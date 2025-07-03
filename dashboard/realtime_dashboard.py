#!/usr/bin/env python3
"""
📊 GoodHunt Real-Time Dashboard
Live Trading Monitoring with Streamlit and Advanced Visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
import queue
import json
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import redis
    import asyncio
    import websockets
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False

from typing import Dict, List, Optional, Any
import logging
import os

class RealtimeDashboard:
    """Real-time trading dashboard with live updates"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = None
        self.data_queue = queue.Queue()
        self.is_running = False
        self.trade_data = []
        self.equity_data = []
        self.performance_metrics = {}
        self.current_positions = {}
        self.market_data = {}
        
        # Initialize Redis connection if available
        if REALTIME_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
                self.redis_client.ping()
                print("✅ Redis connection established")
            except Exception as e:
                print(f"⚠️  Redis connection failed: {e}")
                self.redis_client = None
    
    def start_data_stream(self):
        """Start real-time data streaming"""
        self.is_running = True
        
        # Start background thread for data collection
        data_thread = threading.Thread(target=self._collect_data, daemon=True)
        data_thread.start()
        
        print("🚀 Real-time data stream started")
    
    def stop_data_stream(self):
        """Stop real-time data streaming"""
        self.is_running = False
        print("⏹️  Real-time data stream stopped")
    
    def _collect_data(self):
        """Background data collection from various sources"""
        while self.is_running:
            try:
                # Collect from Redis if available
                if self.redis_client:
                    self._collect_from_redis()
                
                # Collect from database
                self._collect_from_database()
                
                # Simulate real-time data if no sources available
                if not self.redis_client:
                    self._simulate_realtime_data()
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logging.error(f"Data collection error: {e}")
                time.sleep(5)
    
    def _collect_from_redis(self):
        """Collect data from Redis streams"""
        try:
            # Get latest trade data
            trade_data = self.redis_client.get('goodhunt:latest_trade')
            if trade_data:
                trade_json = json.loads(trade_data)
                self.trade_data.append(trade_json)
                
                # Keep only last 1000 trades
                if len(self.trade_data) > 1000:
                    self.trade_data = self.trade_data[-1000:]
            
            # Get equity curve data
            equity_data = self.redis_client.get('goodhunt:equity_curve')
            if equity_data:
                self.equity_data = json.loads(equity_data)
            
            # Get performance metrics
            metrics = self.redis_client.get('goodhunt:performance_metrics')
            if metrics:
                self.performance_metrics = json.loads(metrics)
            
            # Get current positions
            positions = self.redis_client.get('goodhunt:positions')
            if positions:
                self.current_positions = json.loads(positions)
                
        except Exception as e:
            logging.error(f"Redis data collection error: {e}")
    
    def _collect_from_database(self):
        """Collect data from SQLite database"""
        try:
            db_path = "logs/goodhunt_trades.db"
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                
                # Get recent trades
                recent_trades = pd.read_sql_query(
                    "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 100",
                    conn
                )
                
                if not recent_trades.empty:
                    self.trade_data.extend(recent_trades.to_dict('records'))
                
                conn.close()
                
        except Exception as e:
            logging.error(f"Database collection error: {e}")
    
    def _simulate_realtime_data(self):
        """Simulate real-time data for demo purposes"""
        current_time = datetime.now()
        
        # Simulate trade
        if np.random.random() < 0.1:  # 10% chance of trade per second
            trade = {
                'timestamp': current_time.isoformat(),
                'symbol': np.random.choice(['AAPL', 'GOOGL', 'TSLA']),
                'action': np.random.choice(['BUY', 'SELL', 'HOLD']),
                'price': 150 + np.random.normal(0, 5),
                'quantity': np.random.randint(10, 100),
                'pnl': np.random.normal(0, 50),
                'pnl_percent': np.random.normal(0, 2),
                'confidence': np.random.uniform(0.5, 1.0)
            }
            self.trade_data.append(trade)
        
        # Simulate equity curve
        if len(self.equity_data) == 0:
            self.equity_data = [100000]  # Starting balance
        
        # Add small random change
        last_equity = self.equity_data[-1]
        change = np.random.normal(0, 100)
        new_equity = max(0, last_equity + change)
        self.equity_data.append(new_equity)
        
        # Keep only last 1440 points (24 hours if updating every minute)
        if len(self.equity_data) > 1440:
            self.equity_data = self.equity_data[-1440:]
        
        # Update performance metrics
        if len(self.equity_data) > 1:
            total_return = (self.equity_data[-1] - self.equity_data[0]) / self.equity_data[0] * 100
            max_equity = max(self.equity_data)
            current_drawdown = (max_equity - self.equity_data[-1]) / max_equity * 100
            
            self.performance_metrics = {
                'total_return': total_return,
                'current_drawdown': current_drawdown,
                'total_trades': len(self.trade_data),
                'win_rate': np.random.uniform(45, 65),
                'sharpe_ratio': np.random.uniform(0.5, 2.0),
                'profit_factor': np.random.uniform(1.0, 1.8)
            }

def create_main_dashboard():
    """Create the main Streamlit dashboard"""
    st.set_page_config(
        page_title="🦊 GoodHunt Real-Time Dashboard",
        page_icon="🦊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = RealtimeDashboard()
        st.session_state.dashboard.start_data_stream()
    
    dashboard = st.session_state.dashboard
    
    # Dashboard header
    st.title("🦊 GoodHunt Real-Time Trading Dashboard")
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 1, 30, 5)
        
        st.markdown("---")
        
        # System status
        st.subheader("📊 System Status")
        
        if dashboard.redis_client:
            st.success("🟢 Redis Connected")
        else:
            st.warning("🟡 Redis Disconnected")
        
        if dashboard.is_running:
            st.success("🟢 Data Stream Active")
        else:
            st.error("🔴 Data Stream Inactive")
        
        st.info(f"📈 Total Trades: {len(dashboard.trade_data)}")
        st.info(f"💰 Equity Points: {len(dashboard.equity_data)}")
        
        # Manual controls
        st.markdown("---")
        st.subheader("🎮 Manual Controls")
        
        if st.button("🔄 Force Refresh"):
            st.rerun()
        
        if st.button("⏹️ Stop Stream"):
            dashboard.stop_data_stream()
            st.rerun()
        
        if st.button("▶️ Start Stream"):
            dashboard.start_data_stream()
            st.rerun()
    
    # Main dashboard layout
    col1, col2, col3, col4 = st.columns(4)
    
    # Performance metrics cards
    with col1:
        total_return = dashboard.performance_metrics.get('total_return', 0)
        st.metric(
            "Total Return",
            f"{total_return:.2f}%",
            delta=f"{total_return - 5:.2f}%"  # Mock delta
        )
    
    with col2:
        drawdown = dashboard.performance_metrics.get('current_drawdown', 0)
        st.metric(
            "Current Drawdown", 
            f"{drawdown:.2f}%",
            delta=f"{drawdown - 2:.2f}%",
            delta_color="inverse"
        )
    
    with col3:
        win_rate = dashboard.performance_metrics.get('win_rate', 0)
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            delta=f"{win_rate - 50:.1f}%"
        )
    
    with col4:
        sharpe = dashboard.performance_metrics.get('sharpe_ratio', 0)
        st.metric(
            "Sharpe Ratio",
            f"{sharpe:.2f}",
            delta=f"{sharpe - 1:.2f}"
        )
    
    st.markdown("---")
    
    # Charts section
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📈 Equity Curve")
        
        if dashboard.equity_data and len(dashboard.equity_data) > 1:
            equity_df = pd.DataFrame({
                'Time': pd.date_range(
                    end=datetime.now(),
                    periods=len(dashboard.equity_data),
                    freq='1min'
                ),
                'Equity': dashboard.equity_data
            })
            
            fig_equity = px.line(
                equity_df, 
                x='Time', 
                y='Equity',
                title="Portfolio Equity Over Time"
            )
            fig_equity.update_layout(height=400)
            st.plotly_chart(fig_equity, use_container_width=True)
        else:
            st.info("Waiting for equity data...")
    
    with col_right:
        st.subheader("🎯 Current Positions")
        
        if dashboard.current_positions:
            positions_df = pd.DataFrame.from_dict(dashboard.current_positions, orient='index')
            st.dataframe(positions_df, use_container_width=True)
        else:
            st.info("No active positions")
        
        st.subheader("⚡ Recent Actions")
        if dashboard.trade_data:
            recent_trades = dashboard.trade_data[-5:]
            for trade in reversed(recent_trades):
                with st.container():
                    col_a, col_b, col_c = st.columns([1, 1, 1])
                    with col_a:
                        st.text(trade.get('symbol', 'N/A'))
                    with col_b:
                        action_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
                        st.text(f"{action_color.get(trade.get('action', 'HOLD'), '⚪')} {trade.get('action', 'N/A')}")
                    with col_c:
                        pnl = trade.get('pnl', 0)
                        st.text(f"${pnl:+.2f}")
    
    st.markdown("---")
    
    # Detailed analytics
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Trade Analysis", "🧠 AI Insights", "⚠️ Risk Monitor", "🔍 Explainability"])
    
    with tab1:
        st.subheader("Trade Performance Analysis")
        
        if dashboard.trade_data and len(dashboard.trade_data) > 0:
            trades_df = pd.DataFrame(dashboard.trade_data)
            
            # PnL distribution
            col_a, col_b = st.columns(2)
            
            with col_a:
                if 'pnl' in trades_df.columns:
                    fig_pnl = px.histogram(
                        trades_df, 
                        x='pnl',
                        title="PnL Distribution",
                        nbins=20
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)
            
            with col_b:
                if 'action' in trades_df.columns:
                    action_counts = trades_df['action'].value_counts()
                    fig_actions = px.pie(
                        values=action_counts.values,
                        names=action_counts.index,
                        title="Action Distribution"
                    )
                    st.plotly_chart(fig_actions, use_container_width=True)
            
            # Trade timeline
            if 'timestamp' in trades_df.columns:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                trades_df = trades_df.sort_values('timestamp')
                
                fig_timeline = px.scatter(
                    trades_df,
                    x='timestamp',
                    y='pnl',
                    color='action',
                    title="Trade Timeline",
                    hover_data=['symbol', 'price', 'quantity']
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("No trade data available")
    
    with tab2:
        st.subheader("AI Model Insights")
        
        # Model confidence over time
        if dashboard.trade_data:
            trades_df = pd.DataFrame(dashboard.trade_data)
            if 'confidence' in trades_df.columns:
                fig_confidence = px.line(
                    trades_df,
                    y='confidence',
                    title="Model Confidence Over Time"
                )
                st.plotly_chart(fig_confidence, use_container_width=True)
        
        # Feature importance (simulated)
        st.subheader("🔍 Top Features Influencing Decisions")
        features_data = {
            'Feature': ['RSI', 'MACD', 'Volume', 'Sentiment', 'Volatility', 'Momentum'],
            'Importance': [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
        }
        features_df = pd.DataFrame(features_data)
        
        fig_features = px.bar(
            features_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance"
        )
        st.plotly_chart(fig_features, use_container_width=True)
    
    with tab3:
        st.subheader("Risk Monitoring")
        
        # Risk metrics
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            var_95 = np.random.uniform(2000, 5000)
            st.metric("VaR (95%)", f"${var_95:.0f}")
        
        with col_b:
            max_position_size = np.random.uniform(0.1, 0.3)
            st.metric("Max Position Size", f"{max_position_size:.1%}")
        
        with col_c:
            risk_score = np.random.uniform(3, 8)
            st.metric("Risk Score", f"{risk_score:.1f}/10")
        
        # Risk alerts
        st.subheader("⚠️ Risk Alerts")
        alerts = [
            {"level": "🟡 Warning", "message": "Position size approaching limit"},
            {"level": "🟢 Info", "message": "Volatility within normal range"},
            {"level": "🔴 Critical", "message": "Drawdown threshold reached"}
        ]
        
        for alert in alerts:
            st.text(f"{alert['level']}: {alert['message']}")
    
    with tab4:
        st.subheader("Decision Explainability")
        
        # Latest decision explanation
        st.subheader("🔍 Latest Trade Decision")
        
        if dashboard.trade_data:
            latest_trade = dashboard.trade_data[-1] if dashboard.trade_data else {}
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.json({
                    "Action": latest_trade.get('action', 'N/A'),
                    "Symbol": latest_trade.get('symbol', 'N/A'),
                    "Confidence": latest_trade.get('confidence', 0),
                    "Price": latest_trade.get('price', 0)
                })
            
            with col_b:
                # Simulated explanation
                explanation_data = {
                    'Factor': ['RSI Oversold', 'Volume Spike', 'Positive Sentiment', 'Trend Momentum'],
                    'Contribution': [0.3, 0.25, 0.2, 0.15]
                }
                explanation_df = pd.DataFrame(explanation_data)
                
                fig_explanation = px.bar(
                    explanation_df,
                    x='Contribution',
                    y='Factor',
                    orientation='h',
                    title="Decision Factors"
                )
                st.plotly_chart(fig_explanation, use_container_width=True)
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

def create_performance_dashboard():
    """Create performance-focused dashboard"""
    st.title("📊 GoodHunt Performance Dashboard")
    
    # Load performance data
    try:
        # Try to load from saved data
        equity_file = "logs/equity_curve.csv"
        trades_file = "logs/trades.csv"
        
        equity_df = None
        trades_df = None
        
        if os.path.exists(equity_file):
            equity_df = pd.read_csv(equity_file)
        
        if os.path.exists(trades_file):
            trades_df = pd.read_csv(trades_file)
        
        if equity_df is not None and not equity_df.empty:
            # Performance summary
            initial_balance = equity_df.iloc[0]['equity'] if 'equity' in equity_df.columns else 100000
            final_balance = equity_df.iloc[-1]['equity'] if 'equity' in equity_df.columns else 100000
            total_return = (final_balance - initial_balance) / initial_balance * 100
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Initial Balance", f"${initial_balance:,.2f}")
            with col2:
                st.metric("Final Balance", f"${final_balance:,.2f}")
            with col3:
                st.metric("Total Return", f"{total_return:.2f}%")
            with col4:
                max_equity = equity_df['equity'].max() if 'equity' in equity_df.columns else final_balance
                current_dd = (max_equity - final_balance) / max_equity * 100
                st.metric("Max Drawdown", f"{current_dd:.2f}%")
            
            # Equity curve
            st.subheader("📈 Equity Curve")
            if 'equity' in equity_df.columns:
                fig_equity = px.line(equity_df, y='equity', title="Portfolio Equity Over Time")
                st.plotly_chart(fig_equity, use_container_width=True)
        
        if trades_df is not None and not trades_df.empty:
            # Trade analysis
            st.subheader("📊 Trade Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'pnl' in trades_df.columns:
                    winning_trades = len(trades_df[trades_df['pnl'] > 0])
                    total_trades = len(trades_df)
                    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
                    
                    st.metric("Total Trades", total_trades)
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                    
                    # PnL distribution
                    fig_pnl = px.histogram(trades_df, x='pnl', title="PnL Distribution")
                    st.plotly_chart(fig_pnl, use_container_width=True)
            
            with col2:
                if 'action' in trades_df.columns:
                    action_counts = trades_df['action'].value_counts()
                    fig_actions = px.pie(
                        values=action_counts.values,
                        names=action_counts.index,
                        title="Action Distribution"
                    )
                    st.plotly_chart(fig_actions, use_container_width=True)
                
                # Recent trades table
                st.subheader("Recent Trades")
                st.dataframe(trades_df.tail(10), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading performance data: {e}")
        st.info("Please run a backtest first to generate performance data.")

def create_monitoring_dashboard():
    """Create system monitoring dashboard"""
    st.title("🖥️ GoodHunt System Monitor")
    
    # System metrics
    import psutil
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_percent = psutil.cpu_percent(interval=1)
        st.metric("CPU Usage", f"{cpu_percent:.1f}%")
    
    with col2:
        memory = psutil.virtual_memory()
        st.metric("Memory Usage", f"{memory.percent:.1f}%")
    
    with col3:
        disk = psutil.disk_usage('/')
        st.metric("Disk Usage", f"{disk.percent:.1f}%")
    
    with col4:
        process_count = len(psutil.pids())
        st.metric("Active Processes", process_count)
    
    # Log viewer
    st.subheader("📝 Recent Logs")
    
    log_file = "logs/goodhunt_" + datetime.now().strftime('%Y%m%d') + ".log"
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = f.readlines()
            recent_logs = logs[-50:]  # Last 50 lines
            
            for log in reversed(recent_logs):
                log = log.strip()
                if 'ERROR' in log:
                    st.error(log)
                elif 'WARNING' in log:
                    st.warning(log)
                elif 'INFO' in log:
                    st.info(log)
                else:
                    st.text(log)
    else:
        st.info("No log file found")

# Main app runner
def run_dashboard():
    """Run the complete dashboard application"""
    
    # Navigation
    st.sidebar.title("🦊 GoodHunt Dashboard")
    
    page = st.sidebar.selectbox(
        "Choose Dashboard",
        ["🔴 Live Trading", "📊 Performance", "🖥️ System Monitor"]
    )
    
    if page == "🔴 Live Trading":
        create_main_dashboard()
    elif page == "📊 Performance":
        create_performance_dashboard()
    elif page == "🖥️ System Monitor":
        create_monitoring_dashboard()

if __name__ == "__main__":
    run_dashboard()