"""
Configuration Manager for the Solana Token Analysis Agent Swarm.
Handles loading, validating, and providing access to system configuration.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from loguru import logger
from pydantic import BaseModel, Field, ValidationError, validator


class APIConfig(BaseModel):
    """Configuration for external API connections."""
    
    url: str
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    rate_limit: Optional[int] = None  # Requests per minute
    timeout: int = 30  # Timeout in seconds
    retry_attempts: int = 3
    
    @validator('url')
    def url_must_be_valid(cls, v):
        if not v.startswith(('http://', 'https://', 'ws://', 'wss://')):
            raise ValueError('URL must start with http://, https://, ws://, or wss://')
        return v


class SolanaConfig(BaseModel):
    """Configuration for Solana blockchain connection."""
    
    rpc_url: str = "https://api.mainnet-beta.solana.com"
    websocket_url: Optional[str] = None
    commitment: str = "confirmed"
    cluster: str = "mainnet-beta"
    
    @validator('cluster')
    def validate_cluster(cls, v):
        valid_clusters = ['mainnet-beta', 'testnet', 'devnet', 'localnet']
        if v not in valid_clusters:
            raise ValueError(f'Cluster must be one of {valid_clusters}')
        return v


class AgentConfig(BaseModel):
    """Configuration for an individual agent."""
    
    type: str
    enabled: bool = True
    count: int = 1  # Number of instances to create
    params: Dict[str, Any] = {}
    depends_on: List[str] = []


class StorageConfig(BaseModel):
    """Configuration for data storage."""
    
    type: str = "file"  # "file", "sqlite", "postgres"
    path: Optional[str] = None
    connection_string: Optional[str] = None
    cache_ttl: int = 3600  # Cache time-to-live in seconds


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: str = "INFO"
    file_path: Optional[str] = None
    format: str = "{time} | {level} | {message}"
    rotation: str = "1 day"
    retention: str = "1 week"
    
    @validator('level')
    def level_must_be_valid(cls, v):
        valid_levels = ['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()


class NotificationConfig(BaseModel):
    """Configuration for notifications."""
    
    enabled: bool = False
    type: str = "console"  # "console", "email", "telegram", "slack"
    params: Dict[str, Any] = {}


class TokenAnalysisConfig(BaseModel):
    """Configuration for token analysis parameters."""
    
    min_liquidity: float = 10000.0  # Minimum liquidity in USD
    min_holders: int = 50  # Minimum number of unique token holders
    min_transactions: int = 100  # Minimum number of transactions
    confidence_high_threshold: float = 0.8  # Threshold for high confidence
    confidence_medium_threshold: float = 0.5  # Threshold for medium confidence
    blacklisted_tokens: List[str] = []
    blacklisted_developers: List[str] = []
    risk_factors: Dict[str, float] = {
        "low_liquidity": 0.7,
        "few_holders": 0.6,
        "high_concentration": 0.8,
        "suspicious_transactions": 0.9,
        "unknown_developer": 0.5,
        "contract_risk": 0.9
    }


class SwarmConfig(BaseModel):
    """Main configuration for the agent swarm."""
    
    name: str = "Solana Token Analysis Swarm"
    version: str = "0.1.0"
    solana: SolanaConfig = SolanaConfig()
    apis: Dict[str, APIConfig] = {}
    agents: Dict[str, AgentConfig] = {}
    storage: StorageConfig = StorageConfig()
    logging: LoggingConfig = LoggingConfig()
    notifications: NotificationConfig = NotificationConfig()
    token_analysis: TokenAnalysisConfig = TokenAnalysisConfig()
    knowledge_base_path: Optional[str] = None
    max_agents: int = 20


class ConfigManager:
    """
    Manages configuration for the Solana Token Analysis Agent Swarm.
    Handles loading, validating, and providing access to configuration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file.
                If not provided, will look for config.json in the current directory.
        """
        self.config_path = config_path or "config.json"
        self.config: Optional[SwarmConfig] = None
        self.env_prefix = "SOLANA_SWARM_"
    
    def load_config(self) -> SwarmConfig:
        """
        Load and validate the configuration.
        
        Returns:
            The validated configuration.
            
        Raises:
            FileNotFoundError: If the configuration file is not found.
            ValidationError: If the configuration is invalid.
        """
        if not os.path.exists(self.config_path):
            logger.error(f"Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Override with environment variables
            self._apply_env_overrides(config_data)
            
            # Validate and create the config object
            self.config = SwarmConfig(**config_data)
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return self.config
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse configuration file: {str(e)}")
            raise
        except ValidationError as e:
            logger.error(f"Invalid configuration: {str(e)}")
            raise
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> None:
        """
        Apply environment variable overrides to the configuration.
        
        Environment variables should be in the format:
            SOLANA_SWARM_SECTION_KEY=value
            
        For example:
            SOLANA_SWARM_SOLANA_RPC_URL=https://example.com
            
        Args:
            config_data: The configuration data to modify.
        """
        for env_var, value in os.environ.items():
            if env_var.startswith(self.env_prefix):
                # Remove prefix and split into parts
                env_var = env_var[len(self.env_prefix):]
                parts = env_var.lower().split('_')
                
                if len(parts) < 2:
                    continue
                
                # Navigate to the right section of the config
                current = config_data
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    elif not isinstance(current[part], dict):
                        current[part] = {}
                    current = current[part]
                
                # Set the value, converting to the appropriate type
                key = parts[-1]
                current[key] = self._convert_env_value(value)
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool, List, Dict]:
        """
        Convert an environment variable value to the appropriate type.
        
        Args:
            value: The environment variable value.
            
        Returns:
            The converted value.
        """
        # Try to convert to a boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Try to convert to a number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # Try to convert to a list or dict
        if value.startswith('[') and value.endswith(']'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        if value.startswith('{') and value.endswith('}'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        return value
    
    def create_default_config(self, output_path: Optional[str] = None) -> None:
        """
        Create a default configuration file.
        
        Args:
            output_path: Path to save the default configuration.
                If not provided, will use the current config_path.
        """
        output_path = output_path or self.config_path
        
        # Create default config
        default_config = SwarmConfig(
            name="Solana Token Analysis Swarm",
            version="0.1.0",
            solana=SolanaConfig(
                rpc_url="https://api.mainnet-beta.solana.com",
                websocket_url="wss://api.mainnet-beta.solana.com",
                commitment="confirmed",
                cluster="mainnet-beta"
            ),
            apis={
                "solscan": APIConfig(
                    url="https://api.solscan.io",
                    rate_limit=60
                ),
                "coingecko": APIConfig(
                    url="https://api.coingecko.com/api/v3",
                    rate_limit=100
                )
            },
            agents={
                "data_collection": AgentConfig(
                    type="DataCollectionAgent",
                    enabled=True,
                    count=1,
                    params={}
                ),
                "pattern_analysis": AgentConfig(
                    type="PatternAnalysisAgent",
                    enabled=True,
                    count=1,
                    params={},
                    depends_on=["data_collection"]
                ),
                "risk_assessment": AgentConfig(
                    type="RiskAssessmentAgent",
                    enabled=True,
                    count=1,
                    params={},
                    depends_on=["data_collection"]
                ),
                "investment_advisory": AgentConfig(
                    type="InvestmentAdvisoryAgent",
                    enabled=True,
                    count=1,
                    params={},
                    depends_on=["pattern_analysis", "risk_assessment"]
                )
            },
            storage=StorageConfig(
                type="file",
                path="./data"
            ),
            logging=LoggingConfig(
                level="INFO",
                file_path="./logs/swarm.log"
            ),
            notifications=NotificationConfig(
                enabled=True,
                type="console",
                params={}
            ),
            token_analysis=TokenAnalysisConfig(
                min_liquidity=10000.0,
                min_holders=50,
                min_transactions=100,
                confidence_high_threshold=0.8,
                confidence_medium_threshold=0.5,
                blacklisted_tokens=[],
                blacklisted_developers=[],
                risk_factors={
                    "low_liquidity": 0.7,
                    "few_holders": 0.6,
                    "high_concentration": 0.8,
                    "suspicious_transactions": 0.9,
                    "unknown_developer": 0.5,
                    "contract_risk": 0.9
                }
            ),
            knowledge_base_path="./data/knowledge_base.json",
            max_agents=20
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(default_config.json(indent=4))
        
        logger.info(f"Created default configuration at {output_path}")
    
    def get_config(self) -> SwarmConfig:
        """
        Get the current configuration.
        
        Returns:
            The current configuration.
            
        Raises:
            RuntimeError: If configuration has not been loaded yet.
        """
        if self.config is None:
            raise RuntimeError("Configuration has not been loaded yet. Call load_config() first.")
        return self.config
