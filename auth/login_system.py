"""
GoodHunt v3+ Advanced Authentication & Login System
Features:
- User registration and authentication
- Session management
- Role-based access control
- API key management
- Security logging
- User preferences and settings
"""

import hashlib
import secrets
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
from pathlib import Path

class GoodHuntAuth:
    def __init__(self, db_path="auth/users.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(exist_ok=True)
        self.init_database()
        self.setup_logging()
        self.active_sessions = {}
        
    def setup_logging(self):
        """Setup security logging"""
        logging.basicConfig(
            filename='auth/security.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('GoodHuntAuth')
    
    def init_database(self):
        """Initialize user database with proper schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT DEFAULT 'trader',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                failed_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP,
                preferences TEXT DEFAULT '{}',
                api_key TEXT,
                subscription_level TEXT DEFAULT 'basic'
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # API keys table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                key_name TEXT,
                key_hash TEXT,
                permissions TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                last_used TIMESTAMP,
                usage_count INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Security logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT,
                ip_address TEXT,
                user_agent TEXT,
                success BOOLEAN,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """Securely hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        return password_hash.hex(), salt
    
    def register_user(self, username: str, email: str, password: str, 
                     role: str = 'trader', subscription_level: str = 'basic') -> Dict:
        """Register a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', 
                         (username, email))
            if cursor.fetchone():
                return {"success": False, "message": "User already exists"}
            
            # Hash password
            password_hash, salt = self.hash_password(password)
            
            # Generate API key
            api_key = self.generate_api_key()
            
            # Insert user
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, salt, role, 
                                 api_key, subscription_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (username, email, password_hash, salt, role, api_key, subscription_level))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            self.log_security_event(user_id, 'USER_REGISTERED', True, 
                                   f"New user registered: {username}")
            
            return {
                "success": True, 
                "message": "User registered successfully",
                "user_id": user_id,
                "api_key": api_key
            }
            
        except Exception as e:
            self.logger.error(f"Registration error: {str(e)}")
            return {"success": False, "message": "Registration failed"}
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str = None, user_agent: str = None) -> Dict:
        """Authenticate user and create session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user data
            cursor.execute('''
                SELECT id, username, password_hash, salt, role, is_active, 
                       failed_attempts, locked_until, subscription_level
                FROM users WHERE username = ?
            ''', (username,))
            
            user_data = cursor.fetchone()
            if not user_data:
                self.log_security_event(None, 'LOGIN_FAILED', False, 
                                       f"Unknown username: {username}")
                return {"success": False, "message": "Invalid credentials"}
            
            user_id, db_username, stored_hash, salt, role, is_active, failed_attempts, locked_until, sub_level = user_data
            
            # Check if account is locked
            if locked_until and datetime.fromisoformat(locked_until) > datetime.now():
                self.log_security_event(user_id, 'LOGIN_BLOCKED', False, "Account locked")
                return {"success": False, "message": "Account temporarily locked"}
            
            # Check if account is active
            if not is_active:
                self.log_security_event(user_id, 'LOGIN_BLOCKED', False, "Account inactive")
                return {"success": False, "message": "Account inactive"}
            
            # Verify password
            password_hash, _ = self.hash_password(password, salt)
            if password_hash != stored_hash:
                # Increment failed attempts
                failed_attempts += 1
                cursor.execute('''
                    UPDATE users SET failed_attempts = ?, 
                           locked_until = CASE WHEN ? >= 5 THEN ? ELSE locked_until END
                    WHERE id = ?
                ''', (failed_attempts, failed_attempts, 
                      (datetime.now() + timedelta(minutes=30)).isoformat(), user_id))
                conn.commit()
                
                self.log_security_event(user_id, 'LOGIN_FAILED', False, 
                                       f"Invalid password, attempt {failed_attempts}")
                return {"success": False, "message": "Invalid credentials"}
            
            # Reset failed attempts and update last login
            cursor.execute('''
                UPDATE users SET failed_attempts = 0, locked_until = NULL, 
                               last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user_id,))
            
            # Create session
            session_id = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(hours=24)
            
            cursor.execute('''
                INSERT INTO sessions (session_id, user_id, expires_at, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, user_id, expires_at.isoformat(), ip_address, user_agent))
            
            conn.commit()
            conn.close()
            
            self.active_sessions[session_id] = {
                'user_id': user_id,
                'username': db_username,
                'role': role,
                'subscription_level': sub_level,
                'expires_at': expires_at
            }
            
            self.log_security_event(user_id, 'LOGIN_SUCCESS', True, 
                                   f"Successful login from {ip_address}")
            
            return {
                "success": True,
                "session_id": session_id,
                "user_id": user_id,
                "username": db_username,
                "role": role,
                "subscription_level": sub_level,
                "expires_at": expires_at.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
            return {"success": False, "message": "Authentication failed"}
    
    def validate_session(self, session_id: str) -> Dict:
        """Validate session and return user info"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            if session['expires_at'] > datetime.now():
                return {"valid": True, "user": session}
            else:
                del self.active_sessions[session_id]
        
        # Check database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.user_id, u.username, u.role, u.subscription_level, s.expires_at
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_id = ? AND s.is_active = 1
            ''', (session_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and datetime.fromisoformat(result[4]) > datetime.now():
                user_info = {
                    'user_id': result[0],
                    'username': result[1],
                    'role': result[2],
                    'subscription_level': result[3],
                    'expires_at': datetime.fromisoformat(result[4])
                }
                self.active_sessions[session_id] = user_info
                return {"valid": True, "user": user_info}
            
        except Exception as e:
            self.logger.error(f"Session validation error: {str(e)}")
        
        return {"valid": False}
    
    def logout_user(self, session_id: str) -> bool:
        """Logout user and invalidate session"""
        try:
            # Remove from active sessions
            if session_id in self.active_sessions:
                user_id = self.active_sessions[session_id]['user_id']
                del self.active_sessions[session_id]
                
                self.log_security_event(user_id, 'LOGOUT', True, "User logged out")
            
            # Invalidate in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('UPDATE sessions SET is_active = 0 WHERE session_id = ?', 
                         (session_id,))
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Logout error: {str(e)}")
            return False
    
    def generate_api_key(self) -> str:
        """Generate new API key"""
        return f"gh_{secrets.token_urlsafe(32)}"
    
    def create_api_key(self, user_id: int, key_name: str, 
                      permissions: Dict = None, expires_days: int = 90) -> Dict:
        """Create new API key for user"""
        try:
            api_key = self.generate_api_key()
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            expires_at = datetime.now() + timedelta(days=expires_days)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO api_keys (user_id, key_name, key_hash, permissions, expires_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, key_name, key_hash, json.dumps(permissions or {}), 
                  expires_at.isoformat()))
            
            conn.commit()
            conn.close()
            
            self.log_security_event(user_id, 'API_KEY_CREATED', True, 
                                   f"API key created: {key_name}")
            
            return {"success": True, "api_key": api_key, "expires_at": expires_at.isoformat()}
            
        except Exception as e:
            self.logger.error(f"API key creation error: {str(e)}")
            return {"success": False, "message": "Failed to create API key"}
    
    def validate_api_key(self, api_key: str) -> Dict:
        """Validate API key and return user info"""
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT ak.user_id, u.username, u.role, u.subscription_level, 
                       ak.permissions, ak.expires_at, ak.id
                FROM api_keys ak
                JOIN users u ON ak.user_id = u.id
                WHERE ak.key_hash = ? AND ak.is_active = 1 AND u.is_active = 1
            ''', (key_hash,))
            
            result = cursor.fetchone()
            
            if result:
                user_id, username, role, sub_level, permissions, expires_at, key_id = result
                
                if datetime.fromisoformat(expires_at) > datetime.now():
                    # Update usage
                    cursor.execute('''
                        UPDATE api_keys SET usage_count = usage_count + 1,
                               last_used = CURRENT_TIMESTAMP WHERE id = ?
                    ''', (key_id,))
                    conn.commit()
                    
                    conn.close()
                    return {
                        "valid": True,
                        "user_id": user_id,
                        "username": username,
                        "role": role,
                        "subscription_level": sub_level,
                        "permissions": json.loads(permissions)
                    }
            
            conn.close()
            return {"valid": False}
            
        except Exception as e:
            self.logger.error(f"API key validation error: {str(e)}")
            return {"valid": False}
    
    def log_security_event(self, user_id: int, action: str, success: bool, 
                          details: str, ip_address: str = None, user_agent: str = None):
        """Log security events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO security_logs (user_id, action, ip_address, user_agent, 
                                         success, details)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, action, ip_address, user_agent, success, details))
            
            conn.commit()
            conn.close()
            
            # Also log to file
            self.logger.info(f"User {user_id}: {action} - {details} - Success: {success}")
            
        except Exception as e:
            self.logger.error(f"Security logging error: {str(e)}")
    
    def get_user_preferences(self, user_id: int) -> Dict:
        """Get user preferences"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT preferences FROM users WHERE id = ?', (user_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result[0])
            return {}
            
        except Exception as e:
            self.logger.error(f"Get preferences error: {str(e)}")
            return {}
    
    def update_user_preferences(self, user_id: int, preferences: Dict) -> bool:
        """Update user preferences"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('UPDATE users SET preferences = ? WHERE id = ?',
                         (json.dumps(preferences), user_id))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Update preferences error: {str(e)}")
            return False
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Get user statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Basic user info
            cursor.execute('''
                SELECT username, email, role, subscription_level, created_at, last_login
                FROM users WHERE id = ?
            ''', (user_id,))
            user_info = cursor.fetchone()
            
            # Session count
            cursor.execute('''
                SELECT COUNT(*) FROM sessions WHERE user_id = ?
            ''', (user_id,))
            session_count = cursor.fetchone()[0]
            
            # API key count
            cursor.execute('''
                SELECT COUNT(*) FROM api_keys WHERE user_id = ? AND is_active = 1
            ''', (user_id,))
            api_key_count = cursor.fetchone()[0]
            
            # Recent security events
            cursor.execute('''
                SELECT action, success, details, timestamp
                FROM security_logs WHERE user_id = ?
                ORDER BY timestamp DESC LIMIT 10
            ''', (user_id,))
            recent_events = cursor.fetchall()
            
            conn.close()
            
            return {
                "user_info": user_info,
                "session_count": session_count,
                "active_api_keys": api_key_count,
                "recent_events": recent_events
            }
            
        except Exception as e:
            self.logger.error(f"Get user stats error: {str(e)}")
            return {}

# Convenience functions for easy integration
def login(username: str, password: str, ip_address: str = None) -> Dict:
    """Simple login function"""
    auth = GoodHuntAuth()
    return auth.authenticate_user(username, password, ip_address)

def register(username: str, email: str, password: str, 
            role: str = 'trader', subscription: str = 'basic') -> Dict:
    """Simple registration function"""
    auth = GoodHuntAuth()
    return auth.register_user(username, email, password, role, subscription)

def validate_session(session_id: str) -> Dict:
    """Simple session validation"""
    auth = GoodHuntAuth()
    return auth.validate_session(session_id)

def validate_api_key(api_key: str) -> Dict:
    """Simple API key validation"""
    auth = GoodHuntAuth()
    return auth.validate_api_key(api_key)