#!/usr/bin/env python3
"""
💰 GoodHunt Enhanced Transaction Cost Model
Realistic Trading Costs including Spreads, Latency, Slippage, and Margin Requirements
"""

import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import json

@dataclass
class TransactionCostConfig:
    """Configuration for transaction cost modeling"""
    base_commission: float = 0.001  # 0.1% base commission
    min_commission: float = 1.0     # Minimum commission per trade
    spread_multiplier: float = 1.0  # Multiplier for bid-ask spread
    slippage_impact: float = 0.5    # Market impact coefficient
    latency_penalty: float = 0.0001 # Cost per millisecond of latency
    overnight_rate: float = 0.05    # Overnight borrowing rate (annual)
    margin_requirement: float = 0.25 # Margin requirement ratio
    
    # Market-specific parameters
    market_hours: Dict[str, Tuple[time, time]] = None
    asset_class_multipliers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.market_hours is None:
            self.market_hours = {
                'NYSE': (time(9, 30), time(16, 0)),
                'NASDAQ': (time(9, 30), time(16, 0)),
                'LSE': (time(8, 0), time(16, 30)),
                'TSE': (time(9, 0), time(15, 0))
            }
        
        if self.asset_class_multipliers is None:
            self.asset_class_multipliers = {
                'stock': 1.0,
                'etf': 0.8,
                'option': 2.0,
                'future': 0.5,
                'forex': 0.3,
                'crypto': 3.0
            }

class EnhancedTransactionCostModel:
    """Advanced transaction cost model with realistic market conditions"""
    
    def __init__(self, config: TransactionCostConfig = None):
        self.config = config or TransactionCostConfig()
        self.historical_spreads = {}
        self.latency_history = []
        self.volume_impact_cache = {}
        
    def calculate_total_cost(
        self,
        symbol: str,
        price: float,
        quantity: float,
        order_type: str,
        timestamp: datetime,
        market_data: Dict = None,
        asset_class: str = 'stock'
    ) -> Dict[str, float]:
        """Calculate comprehensive transaction costs"""
        
        market_data = market_data or {}
        
        # Base trade value
        trade_value = price * abs(quantity)
        
        # 1. Commission
        commission = self._calculate_commission(trade_value, asset_class)
        
        # 2. Bid-Ask Spread
        spread_cost = self._calculate_spread_cost(
            symbol, price, quantity, market_data, timestamp
        )
        
        # 3. Market Impact (Slippage)
        impact_cost = self._calculate_market_impact(
            symbol, price, quantity, market_data, order_type
        )
        
        # 4. Timing/Latency Cost
        latency_cost = self._calculate_latency_cost(
            trade_value, timestamp, market_data
        )
        
        # 5. Overnight Financing Cost
        overnight_cost = self._calculate_overnight_cost(
            price, quantity, timestamp, asset_class
        )
        
        # 6. Regulatory/Exchange Fees
        regulatory_fees = self._calculate_regulatory_fees(
            trade_value, asset_class
        )
        
        # Total costs
        total_cost = (
            commission + spread_cost + impact_cost + 
            latency_cost + overnight_cost + regulatory_fees
        )
        
        return {
            'commission': commission,
            'spread_cost': spread_cost,
            'market_impact': impact_cost,
            'latency_cost': latency_cost,
            'overnight_cost': overnight_cost,
            'regulatory_fees': regulatory_fees,
            'total_cost': total_cost,
            'cost_per_share': total_cost / abs(quantity) if quantity != 0 else 0,
            'cost_percentage': (total_cost / trade_value * 100) if trade_value > 0 else 0
        }
    
    def _calculate_commission(self, trade_value: float, asset_class: str) -> float:
        """Calculate commission based on trade value and asset class"""
        base_commission = trade_value * self.config.base_commission
        
        # Apply asset class multiplier
        multiplier = self.config.asset_class_multipliers.get(asset_class, 1.0)
        commission = base_commission * multiplier
        
        # Apply minimum commission
        commission = max(commission, self.config.min_commission)
        
        return commission
    
    def _calculate_spread_cost(
        self,
        symbol: str,
        price: float,
        quantity: float,
        market_data: Dict,
        timestamp: datetime
    ) -> float:
        """Calculate bid-ask spread cost"""
        
        # Get spread from market data or estimate
        if 'bid' in market_data and 'ask' in market_data:
            spread = market_data['ask'] - market_data['bid']
        else:
            # Estimate spread based on volatility and price
            volatility = market_data.get('volatility', 0.02)
            spread = self._estimate_spread(price, volatility, timestamp)
        
        # Apply spread multiplier
        spread *= self.config.spread_multiplier
        
        # Spread cost is half the spread (assuming we cross the spread)
        spread_cost = abs(quantity) * spread * 0.5
        
        return spread_cost
    
    def _estimate_spread(self, price: float, volatility: float, timestamp: datetime) -> float:
        """Estimate bid-ask spread based on market conditions"""
        
        # Base spread as percentage of price
        base_spread_pct = 0.001  # 0.1% for liquid stocks
        
        # Adjust for volatility
        volatility_adjustment = 1 + (volatility * 5)  # Higher vol = wider spreads
        
        # Adjust for time of day
        time_of_day = timestamp.time()
        if time_of_day < time(9, 45) or time_of_day > time(15, 45):
            # Wider spreads at market open/close
            time_adjustment = 1.5
        elif time(12, 0) <= time_of_day <= time(14, 0):
            # Lunch time - potentially wider spreads
            time_adjustment = 1.2
        else:
            time_adjustment = 1.0
        
        # Calculate spread
        spread_pct = base_spread_pct * volatility_adjustment * time_adjustment
        spread = price * spread_pct
        
        return spread
    
    def _calculate_market_impact(
        self,
        symbol: str,
        price: float,
        quantity: float,
        market_data: Dict,
        order_type: str
    ) -> float:
        """Calculate market impact (temporary and permanent)"""
        
        # Get average daily volume
        avg_volume = market_data.get('avg_volume', 1000000)
        current_volume = market_data.get('volume', avg_volume)
        
        # Calculate volume participation rate
        participation_rate = abs(quantity) / max(current_volume, 1)
        
        # Impact model: square root law
        impact_coefficient = self.config.slippage_impact
        
        # Market impact (bps)
        if order_type.upper() == 'MARKET':
            # Market orders have higher impact
            impact_bps = impact_coefficient * np.sqrt(participation_rate) * 100
        else:
            # Limit orders have lower impact
            impact_bps = impact_coefficient * np.sqrt(participation_rate) * 50
        
        # Convert to dollar impact
        impact_cost = abs(quantity) * price * (impact_bps / 10000)
        
        return impact_cost
    
    def _calculate_latency_cost(
        self,
        trade_value: float,
        timestamp: datetime,
        market_data: Dict
    ) -> float:
        """Calculate cost due to execution latency"""
        
        # Simulate latency (in practice, measure actual latency)
        base_latency_ms = 50  # 50ms base latency
        network_latency = np.random.exponential(20)  # Additional network latency
        total_latency_ms = base_latency_ms + network_latency
        
        # Store latency for monitoring
        self.latency_history.append(total_latency_ms)
        if len(self.latency_history) > 1000:
            self.latency_history = self.latency_history[-1000:]
        
        # Cost increases with latency and volatility
        volatility = market_data.get('volatility', 0.02)
        latency_cost = (
            trade_value * 
            self.config.latency_penalty * 
            total_latency_ms * 
            volatility
        )
        
        return latency_cost
    
    def _calculate_overnight_cost(
        self,
        price: float,
        quantity: float,
        timestamp: datetime,
        asset_class: str
    ) -> float:
        """Calculate overnight financing/borrowing costs"""
        
        # Only apply to positions held overnight
        if asset_class not in ['stock', 'etf']:
            return 0.0
        
        # Assume position held overnight (in practice, track actual holding period)
        position_value = abs(price * quantity)
        
        # Margin requirement
        margin_required = position_value * self.config.margin_requirement
        borrowed_amount = position_value - margin_required
        
        if borrowed_amount > 0:
            # Daily borrowing cost
            daily_rate = self.config.overnight_rate / 365
            overnight_cost = borrowed_amount * daily_rate
        else:
            overnight_cost = 0.0
        
        return overnight_cost
    
    def _calculate_regulatory_fees(self, trade_value: float, asset_class: str) -> float:
        """Calculate regulatory and exchange fees"""
        
        fees = 0.0
        
        if asset_class == 'stock':
            # SEC fee (sell-side only, simplified here)
            sec_fee = trade_value * 0.0000051  # Current SEC fee rate
            
            # FINRA Trading Activity Fee
            finra_fee = min(trade_value * 0.000145, 7.27)  # Capped at $7.27
            
            fees = sec_fee + finra_fee
            
        elif asset_class == 'option':
            # Options regulatory fee per contract
            contracts = trade_value / 100  # Assuming $100 per contract average
            fees = contracts * 0.04  # $0.04 per contract
            
        elif asset_class == 'future':
            # Exchange fees for futures
            fees = 2.50  # Fixed fee per contract
        
        return fees
    
    def get_cost_breakdown_report(
        self,
        symbol: str,
        price: float,
        quantity: float,
        order_type: str,
        timestamp: datetime,
        market_data: Dict = None,
        asset_class: str = 'stock'
    ) -> str:
        """Generate detailed cost breakdown report"""
        
        costs = self.calculate_total_cost(
            symbol, price, quantity, order_type, timestamp, market_data, asset_class
        )
        
        trade_value = price * abs(quantity)
        
        report = f"""
📊 GoodHunt Transaction Cost Analysis
═══════════════════════════════════════

Trade Details:
• Symbol: {symbol}
• Price: ${price:.2f}
• Quantity: {quantity:,.0f}
• Order Type: {order_type}
• Asset Class: {asset_class.upper()}
• Trade Value: ${trade_value:,.2f}
• Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Cost Breakdown:
• Commission: ${costs['commission']:.2f} ({costs['commission']/trade_value*100:.3f}%)
• Spread Cost: ${costs['spread_cost']:.2f} ({costs['spread_cost']/trade_value*100:.3f}%)
• Market Impact: ${costs['market_impact']:.2f} ({costs['market_impact']/trade_value*100:.3f}%)
• Latency Cost: ${costs['latency_cost']:.2f} ({costs['latency_cost']/trade_value*100:.3f}%)
• Overnight Cost: ${costs['overnight_cost']:.2f} ({costs['overnight_cost']/trade_value*100:.3f}%)
• Regulatory Fees: ${costs['regulatory_fees']:.2f} ({costs['regulatory_fees']/trade_value*100:.3f}%)

Total Transaction Cost: ${costs['total_cost']:.2f} ({costs['cost_percentage']:.3f}%)
Cost per Share: ${costs['cost_per_share']:.4f}

Performance Impact:
• Break-even Move: {costs['cost_per_share']/price*100:.3f}%
• Annual Impact (100 trades): {costs['cost_percentage']*100:.1f}%
"""
        
        return report
    
    def optimize_execution_timing(
        self,
        symbol: str,
        target_quantity: float,
        max_participation: float = 0.1,
        time_horizon_minutes: int = 60
    ) -> List[Dict]:
        """Suggest optimal execution schedule to minimize costs"""
        
        # TWAP-style execution schedule
        num_slices = min(time_horizon_minutes // 5, 12)  # 5-minute slices, max 12
        slice_size = target_quantity / num_slices
        
        schedule = []
        
        for i in range(num_slices):
            execution_time = datetime.now() + timedelta(minutes=i*5)
            
            # Adjust slice size based on time of day
            time_factor = self._get_time_factor(execution_time.time())
            adjusted_size = slice_size * time_factor
            
            schedule.append({
                'execution_time': execution_time,
                'quantity': adjusted_size,
                'order_type': 'LIMIT',
                'urgency': 'LOW' if time_factor < 1.2 else 'HIGH'
            })
        
        return schedule
    
    def _get_time_factor(self, execution_time: time) -> float:
        """Get execution difficulty factor based on time of day"""
        
        if time(9, 30) <= execution_time <= time(10, 0):
            return 1.5  # High volatility at open
        elif time(15, 30) <= execution_time <= time(16, 0):
            return 1.4  # High volatility at close
        elif time(12, 0) <= execution_time <= time(14, 0):
            return 1.1  # Lunch time - moderate liquidity
        else:
            return 1.0  # Normal trading hours
    
    def simulate_execution_costs(
        self,
        trades: List[Dict],
        market_scenarios: List[Dict]
    ) -> pd.DataFrame:
        """Simulate execution costs across different market scenarios"""
        
        results = []
        
        for scenario in market_scenarios:
            scenario_name = scenario.get('name', 'Unknown')
            
            for trade in trades:
                # Apply scenario conditions
                market_data = scenario.get('market_data', {})
                
                costs = self.calculate_total_cost(
                    symbol=trade['symbol'],
                    price=trade['price'],
                    quantity=trade['quantity'],
                    order_type=trade.get('order_type', 'MARKET'),
                    timestamp=trade.get('timestamp', datetime.now()),
                    market_data=market_data,
                    asset_class=trade.get('asset_class', 'stock')
                )
                
                result = {
                    'scenario': scenario_name,
                    'symbol': trade['symbol'],
                    'trade_value': trade['price'] * abs(trade['quantity']),
                    **costs
                }
                results.append(result)
        
        return pd.DataFrame(results)

class BrokerSpecificCosts:
    """Broker-specific cost models"""
    
    @staticmethod
    def interactive_brokers_model() -> TransactionCostConfig:
        """Interactive Brokers cost structure"""
        return TransactionCostConfig(
            base_commission=0.005,  # $0.005 per share
            min_commission=1.0,     # $1 minimum
            spread_multiplier=1.0,
            slippage_impact=0.3,
            overnight_rate=0.0583   # Current IB rate
        )
    
    @staticmethod
    def alpaca_model() -> TransactionCostConfig:
        """Alpaca commission-free model"""
        return TransactionCostConfig(
            base_commission=0.0,    # Commission-free
            min_commission=0.0,
            spread_multiplier=1.2,  # Wider spreads due to payment for order flow
            slippage_impact=0.4,
            overnight_rate=0.0575
        )
    
    @staticmethod
    def binance_model() -> TransactionCostConfig:
        """Binance crypto trading costs"""
        return TransactionCostConfig(
            base_commission=0.001,  # 0.1% maker/taker fee
            min_commission=0.0,
            spread_multiplier=2.0,  # Crypto spreads can be wide
            slippage_impact=1.0,    # High impact in crypto
            overnight_rate=0.0,     # No overnight costs in crypto
            asset_class_multipliers={'crypto': 1.0}
        )

# Utility functions
def estimate_daily_transaction_costs(
    daily_trades: int,
    avg_trade_size: float,
    avg_price: float,
    cost_model: EnhancedTransactionCostModel
) -> Dict[str, float]:
    """Estimate daily transaction costs for a trading strategy"""
    
    total_volume = daily_trades * avg_trade_size * avg_price
    
    # Sample cost calculation
    sample_cost = cost_model.calculate_total_cost(
        symbol='SAMPLE',
        price=avg_price,
        quantity=avg_trade_size,
        order_type='MARKET',
        timestamp=datetime.now()
    )
    
    daily_costs = {
        'total_cost': sample_cost['total_cost'] * daily_trades,
        'cost_percentage': sample_cost['cost_percentage'],
        'commission_total': sample_cost['commission'] * daily_trades,
        'spread_cost_total': sample_cost['spread_cost'] * daily_trades,
        'impact_cost_total': sample_cost['market_impact'] * daily_trades
    }
    
    # Annualized estimates (assuming 252 trading days)
    daily_costs['annual_cost'] = daily_costs['total_cost'] * 252
    daily_costs['annual_drag'] = daily_costs['cost_percentage'] * daily_trades * 252
    
    return daily_costs

# Example usage and testing
if __name__ == "__main__":
    # Create enhanced cost model
    config = TransactionCostConfig()
    cost_model = EnhancedTransactionCostModel(config)
    
    # Example trade
    trade_costs = cost_model.calculate_total_cost(
        symbol='AAPL',
        price=150.0,
        quantity=1000,
        order_type='MARKET',
        timestamp=datetime.now(),
        market_data={
            'volatility': 0.025,
            'volume': 50000000,
            'avg_volume': 45000000
        }
    )
    
    print("💰 GoodHunt Enhanced Transaction Cost Model")
    print(f"📊 Sample Trade Cost: ${trade_costs['total_cost']:.2f}")
    print(f"📈 Cost Percentage: {trade_costs['cost_percentage']:.3f}%")
    
    # Generate detailed report
    report = cost_model.get_cost_breakdown_report(
        symbol='AAPL',
        price=150.0,
        quantity=1000,
        order_type='MARKET',
        timestamp=datetime.now(),
        market_data={'volatility': 0.025, 'volume': 50000000}
    )
    
    print(report)