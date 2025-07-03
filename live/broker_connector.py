#!/usr/bin/env python3
"""
Broker Connector for GoodHunt v3+ Live Trading
==============================================
Multi-broker integration for live trading execution
"""

import asyncio
import logging
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import threading
import queue
import requests
import websocket
import ccxt
from enum import Enum

logger = logging.getLogger('GoodHunt.BrokerConnector')

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Order data structure"""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    amount: float
    price: Optional[float]
    status: OrderStatus
    filled: float = 0.0
    remaining: float = 0.0
    timestamp: datetime = None
    info: Dict = None

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    side: OrderSide
    amount: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime

class BrokerInterface(ABC):
    """Abstract interface for broker implementations"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker"""
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, side: OrderSide, amount: float, 
                         order_type: OrderType = OrderType.MARKET, 
                         price: float = None) -> Order:
        """Place an order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str) -> Order:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get all positions"""
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        pass

class PaperTradingBroker(BrokerInterface):
    """Paper trading broker for simulation"""
    
    def __init__(self, initial_balance: float = 100000.0):
        self.connected = False
        self.initial_balance = initial_balance
        self.balance = {"USD": initial_balance}
        self.positions = {}
        self.orders = {}
        self.order_counter = 0
        self.market_data = {}
        
    async def connect(self) -> bool:
        """Connect to paper trading"""
        self.connected = True
        logger.info("✅ Connected to paper trading broker")
        return True
    
    async def disconnect(self):
        """Disconnect from paper trading"""
        self.connected = False
        logger.info("🔌 Disconnected from paper trading broker")
    
    async def place_order(self, symbol: str, side: OrderSide, amount: float,
                         order_type: OrderType = OrderType.MARKET,
                         price: float = None) -> Order:
        """Place a paper trading order"""
        try:
            self.order_counter += 1
            order_id = f"paper_{self.order_counter}_{int(time.time())}"
            
            # Simulate market price if not provided
            if price is None:
                price = self.get_market_price(symbol)
            
            order = Order(
                id=order_id,
                symbol=symbol,
                side=side,
                type=order_type,
                amount=amount,
                price=price,
                status=OrderStatus.PENDING,
                timestamp=datetime.now()
            )
            
            # Simulate immediate execution for market orders
            if order_type == OrderType.MARKET:
                execution_price = price * (1.001 if side == OrderSide.BUY else 0.999)  # Simulate slippage
                await self.execute_order(order, execution_price)
            
            self.orders[order_id] = order
            logger.info(f"📝 Paper order placed: {order_id} {side.value} {amount} {symbol} @ {price}")
            return order
            
        except Exception as e:
            logger.error(f"❌ Failed to place paper order: {e}")
            raise
    
    async def execute_order(self, order: Order, execution_price: float):
        """Execute a paper trading order"""
        try:
            # Update position
            position_key = order.symbol
            if position_key not in self.positions:
                self.positions[position_key] = Position(
                    symbol=order.symbol,
                    side=order.side,
                    amount=0.0,
                    entry_price=execution_price,
                    current_price=execution_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    timestamp=datetime.now()
                )
            
            position = self.positions[position_key]
            
            if order.side == OrderSide.BUY:
                # Calculate new average price
                total_cost = position.amount * position.entry_price + order.amount * execution_price
                total_amount = position.amount + order.amount
                position.entry_price = total_cost / total_amount if total_amount > 0 else execution_price
                position.amount = total_amount
                position.side = OrderSide.BUY
                
                # Update balance
                cost = order.amount * execution_price
                self.balance["USD"] -= cost
                
            else:  # SELL
                if position.amount >= order.amount:
                    # Calculate realized PnL
                    realized_pnl = (execution_price - position.entry_price) * order.amount
                    position.realized_pnl += realized_pnl
                    position.amount -= order.amount
                    
                    # Update balance
                    proceeds = order.amount * execution_price
                    self.balance["USD"] += proceeds
                    
                    if position.amount <= 0:
                        position.amount = 0
                        position.side = OrderSide.BUY  # Reset
                else:
                    logger.warning(f"⚠️  Insufficient position to sell {order.amount} {order.symbol}")
                    order.status = OrderStatus.REJECTED
                    return
            
            order.status = OrderStatus.CLOSED
            order.filled = order.amount
            order.remaining = 0.0
            
            logger.info(f"✅ Paper order executed: {order.id} at {execution_price}")
            
        except Exception as e:
            logger.error(f"❌ Failed to execute paper order: {e}")
            order.status = OrderStatus.REJECTED
    
    def get_market_price(self, symbol: str) -> float:
        """Get simulated market price"""
        # Return simulated price (in production, connect to real data feed)
        base_price = 100.0
        if symbol in self.market_data:
            return self.market_data[symbol]
        return base_price
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a paper trading order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == OrderStatus.OPEN:
                order.status = OrderStatus.CANCELED
                logger.info(f"❌ Paper order canceled: {order_id}")
                return True
        return False
    
    async def get_order(self, order_id: str) -> Order:
        """Get paper trading order"""
        return self.orders.get(order_id)
    
    async def get_positions(self) -> List[Position]:
        """Get all paper trading positions"""
        return [pos for pos in self.positions.values() if pos.amount > 0]
    
    async def get_balance(self) -> Dict[str, float]:
        """Get paper trading balance"""
        return self.balance.copy()

class CCXTBroker(BrokerInterface):
    """CCXT-based broker for crypto exchanges"""
    
    def __init__(self, exchange_id: str, api_key: str, secret: str, sandbox: bool = True):
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.secret = secret
        self.sandbox = sandbox
        self.exchange = None
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to exchange via CCXT"""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.secret,
                'sandbox': self.sandbox,
                'enableRateLimit': True,
            })
            
            # Test connection
            await self.exchange.load_markets()
            balance = await self.exchange.fetch_balance()
            
            self.connected = True
            logger.info(f"✅ Connected to {self.exchange_id} exchange")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to {self.exchange_id}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from exchange"""
        if self.exchange:
            await self.exchange.close()
        self.connected = False
        logger.info(f"🔌 Disconnected from {self.exchange_id}")
    
    async def place_order(self, symbol: str, side: OrderSide, amount: float,
                         order_type: OrderType = OrderType.MARKET,
                         price: float = None) -> Order:
        """Place order on exchange"""
        try:
            order_params = {}
            if order_type == OrderType.LIMIT and price is None:
                raise ValueError("Price required for limit orders")
            
            result = await self.exchange.create_order(
                symbol, order_type.value, side.value, amount, price, order_params
            )
            
            order = Order(
                id=result['id'],
                symbol=symbol,
                side=side,
                type=order_type,
                amount=amount,
                price=price,
                status=OrderStatus.OPEN,
                filled=result.get('filled', 0),
                remaining=result.get('remaining', amount),
                timestamp=datetime.now(),
                info=result
            )
            
            logger.info(f"📝 Order placed on {self.exchange_id}: {order.id}")
            return order
            
        except Exception as e:
            logger.error(f"❌ Failed to place order on {self.exchange_id}: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on exchange"""
        try:
            await self.exchange.cancel_order(order_id)
            logger.info(f"❌ Order canceled on {self.exchange_id}: {order_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to cancel order: {e}")
            return False
    
    async def get_order(self, order_id: str) -> Order:
        """Get order from exchange"""
        try:
            result = await self.exchange.fetch_order(order_id)
            return Order(
                id=result['id'],
                symbol=result['symbol'],
                side=OrderSide(result['side']),
                type=OrderType(result['type']),
                amount=result['amount'],
                price=result['price'],
                status=OrderStatus(result['status']),
                filled=result['filled'],
                remaining=result['remaining'],
                timestamp=datetime.fromtimestamp(result['timestamp'] / 1000),
                info=result
            )
        except Exception as e:
            logger.error(f"❌ Failed to get order: {e}")
            return None
    
    async def get_positions(self) -> List[Position]:
        """Get positions from exchange"""
        try:
            positions = await self.exchange.fetch_positions()
            result = []
            
            for pos in positions:
                if pos['contracts'] > 0:
                    result.append(Position(
                        symbol=pos['symbol'],
                        side=OrderSide.BUY if pos['side'] == 'long' else OrderSide.SELL,
                        amount=pos['contracts'],
                        entry_price=pos['entryPrice'],
                        current_price=pos['markPrice'],
                        unrealized_pnl=pos['unrealizedPnl'],
                        realized_pnl=pos['realizedPnl'],
                        timestamp=datetime.now()
                    ))
            
            return result
        except Exception as e:
            logger.error(f"❌ Failed to get positions: {e}")
            return []
    
    async def get_balance(self) -> Dict[str, float]:
        """Get balance from exchange"""
        try:
            balance = await self.exchange.fetch_balance()
            return {currency: info['free'] for currency, info in balance.items() 
                   if isinstance(info, dict) and 'free' in info}
        except Exception as e:
            logger.error(f"❌ Failed to get balance: {e}")
            return {}

class BrokerConnector:
    """Main broker connector managing multiple broker connections"""
    
    def __init__(self):
        self.brokers = {}
        self.active_broker = None
        self.order_queue = queue.Queue()
        self.position_cache = {}
        self.balance_cache = {}
        self.last_update = None
        self.running = False
        
    def add_broker(self, name: str, broker: BrokerInterface):
        """Add a broker to the connector"""
        self.brokers[name] = broker
        logger.info(f"✅ Added broker: {name}")
    
    async def connect_broker(self, name: str) -> bool:
        """Connect to a specific broker"""
        if name not in self.brokers:
            logger.error(f"❌ Broker not found: {name}")
            return False
        
        broker = self.brokers[name]
        if await broker.connect():
            self.active_broker = name
            logger.info(f"✅ Connected to active broker: {name}")
            return True
        return False
    
    async def disconnect_all(self):
        """Disconnect from all brokers"""
        for name, broker in self.brokers.items():
            try:
                await broker.disconnect()
            except Exception as e:
                logger.error(f"❌ Error disconnecting {name}: {e}")
        self.active_broker = None
    
    async def place_order(self, symbol: str, side: str, amount: float,
                         order_type: str = "market", price: float = None) -> Order:
        """Place order using active broker"""
        if not self.active_broker:
            raise Exception("No active broker connected")
        
        broker = self.brokers[self.active_broker]
        return await broker.place_order(
            symbol, OrderSide(side), amount, OrderType(order_type), price
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order using active broker"""
        if not self.active_broker:
            return False
        
        broker = self.brokers[self.active_broker]
        return await broker.cancel_order(order_id)
    
    async def get_positions(self) -> List[Position]:
        """Get all positions from active broker"""
        if not self.active_broker:
            return []
        
        broker = self.brokers[self.active_broker]
        positions = await broker.get_positions()
        self.position_cache = {pos.symbol: pos for pos in positions}
        return positions
    
    async def get_balance(self) -> Dict[str, float]:
        """Get balance from active broker"""
        if not self.active_broker:
            return {}
        
        broker = self.brokers[self.active_broker]
        balance = await broker.get_balance()
        self.balance_cache = balance
        return balance
    
    async def close_all_positions(self):
        """Close all open positions"""
        try:
            positions = await self.get_positions()
            for position in positions:
                if position.amount > 0:
                    # Close position by selling
                    await self.place_order(
                        position.symbol,
                        "sell" if position.side == OrderSide.BUY else "buy",
                        position.amount,
                        "market"
                    )
            logger.info("✅ All positions closed")
        except Exception as e:
            logger.error(f"❌ Failed to close positions: {e}")
    
    def start_monitoring(self):
        """Start background monitoring"""
        self.running = True
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        logger.info("🔍 Started broker monitoring")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.running = False
        logger.info("⏹️  Stopped broker monitoring")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                if self.active_broker:
                    # Update positions and balance cache
                    asyncio.run(self.get_positions())
                    asyncio.run(self.get_balance())
                    self.last_update = datetime.now()
                
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                time.sleep(10)
    
    def get_broker_status(self) -> Dict[str, Any]:
        """Get status of all brokers"""
        status = {
            "active_broker": self.active_broker,
            "connected_brokers": [],
            "total_brokers": len(self.brokers),
            "last_update": self.last_update,
            "cached_positions": len(self.position_cache),
            "balance_currencies": len(self.balance_cache)
        }
        
        for name, broker in self.brokers.items():
            if hasattr(broker, 'connected') and broker.connected:
                status["connected_brokers"].append(name)
        
        return status

# Factory function to create brokers
def create_broker(broker_type: str, **kwargs) -> BrokerInterface:
    """Factory function to create broker instances"""
    if broker_type == "paper":
        return PaperTradingBroker(kwargs.get("initial_balance", 100000.0))
    elif broker_type == "ccxt":
        return CCXTBroker(
            kwargs.get("exchange_id"),
            kwargs.get("api_key"),
            kwargs.get("secret"),
            kwargs.get("sandbox", True)
        )
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")