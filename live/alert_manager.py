#!/usr/bin/env python3
"""
Alert Manager for GoodHunt v3+
==============================
Comprehensive alert and notification system
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger('GoodHunt.AlertManager')

class AlertType(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    LOG = "log"

@dataclass
class Alert:
    """Alert data structure"""
    timestamp: datetime
    alert_type: AlertType
    title: str
    message: str
    symbol: str = ""
    data: Dict = None

class AlertManager:
    """
    Comprehensive alert and notification management system
    """
    
    def __init__(self):
        self.alerts_history = []
        self.config = {}
        self.enabled_channels = set()
        
        logger.info("🔔 Alert Manager initialized")
    
    def configure(self, config: Dict[str, Any]):
        """Configure alert channels"""
        self.config = config
        
        # Enable channels based on config
        if config.get('email', {}).get('enabled', False):
            self.enabled_channels.add(AlertChannel.EMAIL)
        if config.get('slack', {}).get('enabled', False):
            self.enabled_channels.add(AlertChannel.SLACK)
        if config.get('discord', {}).get('enabled', False):
            self.enabled_channels.add(AlertChannel.DISCORD)
        
        # Log channel is always enabled
        self.enabled_channels.add(AlertChannel.LOG)
        
        logger.info(f"📡 Configured alert channels: {[c.value for c in self.enabled_channels]}")
    
    def send_alert(self, alert_type: AlertType, title: str, message: str, 
                   symbol: str = "", data: Dict = None):
        """Send alert through all configured channels"""
        try:
            alert = Alert(
                timestamp=datetime.now(),
                alert_type=alert_type,
                title=title,
                message=message,
                symbol=symbol,
                data=data or {}
            )
            
            # Store alert
            self.alerts_history.append(alert)
            
            # Send through all enabled channels
            for channel in self.enabled_channels:
                try:
                    if channel == AlertChannel.EMAIL:
                        self._send_email_alert(alert)
                    elif channel == AlertChannel.SLACK:
                        self._send_slack_alert(alert)
                    elif channel == AlertChannel.DISCORD:
                        self._send_discord_alert(alert)
                    elif channel == AlertChannel.LOG:
                        self._log_alert(alert)
                except Exception as e:
                    logger.error(f"❌ Failed to send alert via {channel.value}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send alert: {e}")
    
    def _send_email_alert(self, alert: Alert):
        """Send alert via email"""
        try:
            email_config = self.config.get('email', {})
            
            msg = MIMEMultipart()
            msg['From'] = email_config.get('from_email', '')
            msg['To'] = ', '.join(email_config.get('to_emails', []))
            msg['Subject'] = f"[GoodHunt] {alert.alert_type.value.upper()}: {alert.title}"
            
            body = f"""
GoodHunt Trading Alert

Type: {alert.alert_type.value.upper()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Symbol: {alert.symbol}
Title: {alert.title}

Message:
{alert.message}

Additional Data:
{json.dumps(alert.data, indent=2) if alert.data else 'None'}

---
This is an automated message from GoodHunt Trading System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config.get('smtp_server', ''), email_config.get('smtp_port', 587))
            server.starttls()
            server.login(email_config.get('username', ''), email_config.get('password', ''))
            server.send_message(msg)
            server.quit()
            
            logger.debug(f"📧 Email alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"❌ Email alert failed: {e}")
    
    def _send_slack_alert(self, alert: Alert):
        """Send alert via Slack webhook"""
        try:
            slack_config = self.config.get('slack', {})
            webhook_url = slack_config.get('webhook_url', '')
            
            # Color based on alert type
            color_map = {
                AlertType.INFO: "good",
                AlertType.WARNING: "warning", 
                AlertType.ERROR: "danger",
                AlertType.CRITICAL: "danger"
            }
            
            payload = {
                "channel": slack_config.get('channel', '#trading'),
                "username": "GoodHunt Bot",
                "icon_emoji": ":robot_face:",
                "attachments": [{
                    "color": color_map.get(alert.alert_type, "warning"),
                    "title": f"{alert.alert_type.value.upper()}: {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Symbol", "value": alert.symbol, "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime('%H:%M:%S'), "short": True}
                    ],
                    "footer": "GoodHunt Trading System",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.debug(f"💬 Slack alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"❌ Slack alert failed: {e}")
    
    def _send_discord_alert(self, alert: Alert):
        """Send alert via Discord webhook"""
        try:
            discord_config = self.config.get('discord', {})
            webhook_url = discord_config.get('webhook_url', '')
            
            # Color based on alert type
            color_map = {
                AlertType.INFO: 0x00ff00,
                AlertType.WARNING: 0xffff00,
                AlertType.ERROR: 0xff0000,
                AlertType.CRITICAL: 0xff0000
            }
            
            embed = {
                "title": f"{alert.alert_type.value.upper()}: {alert.title}",
                "description": alert.message,
                "color": color_map.get(alert.alert_type, 0xffff00),
                "fields": [
                    {"name": "Symbol", "value": alert.symbol or "N/A", "inline": True},
                    {"name": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "inline": True}
                ],
                "footer": {"text": "GoodHunt Trading System"},
                "timestamp": alert.timestamp.isoformat()
            }
            
            payload = {"embeds": [embed]}
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.debug(f"🎮 Discord alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"❌ Discord alert failed: {e}")
    
    def _log_alert(self, alert: Alert):
        """Log alert to system logger"""
        log_message = f"[{alert.symbol}] {alert.title}: {alert.message}"
        
        if alert.alert_type == AlertType.CRITICAL:
            logger.critical(f"🚨 {log_message}")
        elif alert.alert_type == AlertType.ERROR:
            logger.error(f"❌ {log_message}")
        elif alert.alert_type == AlertType.WARNING:
            logger.warning(f"⚠️  {log_message}")
        else:
            logger.info(f"ℹ️  {log_message}")
    
    # Convenience methods for different alert types
    def info(self, title: str, message: str, symbol: str = "", data: Dict = None):
        """Send info alert"""
        self.send_alert(AlertType.INFO, title, message, symbol, data)
    
    def warning(self, title: str, message: str, symbol: str = "", data: Dict = None):
        """Send warning alert"""
        self.send_alert(AlertType.WARNING, title, message, symbol, data)
    
    def error(self, title: str, message: str, symbol: str = "", data: Dict = None):
        """Send error alert"""
        self.send_alert(AlertType.ERROR, title, message, symbol, data)
    
    def critical(self, title: str, message: str, symbol: str = "", data: Dict = None):
        """Send critical alert"""
        self.send_alert(AlertType.CRITICAL, title, message, symbol, data)
    
    def get_recent_alerts(self, count: int = 20) -> List[Dict]:
        """Get recent alerts"""
        recent = self.alerts_history[-count:] if len(self.alerts_history) > count else self.alerts_history
        return [
            {
                "timestamp": alert.timestamp.isoformat(),
                "type": alert.alert_type.value,
                "title": alert.title,
                "message": alert.message,
                "symbol": alert.symbol,
                "data": alert.data
            }
            for alert in recent
        ]