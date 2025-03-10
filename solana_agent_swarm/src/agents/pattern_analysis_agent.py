"""
Pattern Analysis Agent for the Solana Token Analysis Agent Swarm.
Analyzes token transactions and market behavior to identify patterns
that may indicate high-performing tokens.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from ..core.agent_base import Agent, AgentStatus, Message, MessageType
from ..core.knowledge_base import EntryType, KnowledgeBase


class PatternType:
    """Constants for different pattern types detected."""
    VOLUME_SPIKE = "volume_spike"
    BUY_PRESSURE = "buy_pressure"
    WHALE_ENTRY = "whale_entry"
    HOLDER_GROWTH = "holder_growth"
    PRICE_BREAKOUT = "price_breakout"
    LOW_VOLATILITY = "low_volatility"
    ACCUMULATION = "accumulation"
    DEV_ACTIVITY = "dev_activity"


class PatternSeverity:
    """Constants for pattern significance levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PatternAnalysisAgent(Agent):
    """
    Agent responsible for analyzing token transaction patterns.
    Identifies patterns that may indicate high-performing tokens.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Pattern Analysis Agent.
        
        Args:
            agent_id: Optional agent ID. If not provided, one will be generated.
            knowledge_base: Optional knowledge base instance to use.
                If not provided, the agent will expect to receive one
                from the orchestrator after initialization.
            config: Optional configuration for pattern detection thresholds.
        """
        agent_id = agent_id or f"pattern-analyzer-{str(uuid.uuid4())[:8]}"
        super().__init__(agent_id=agent_id, agent_type="PatternAnalysisAgent")
        
        self.knowledge_base = knowledge_base
        
        # Default configuration - can be overridden by provided config
        self.config = {
            "volume_spike_threshold": 2.5,  # Multiple of average volume
            "buy_pressure_threshold": 0.65,  # Ratio of buys to total transactions
            "whale_entry_min_percentage": 5.0,  # Minimum percentage for whale detection
            "holder_growth_threshold": 10.0,  # Percentage growth in holders
            "price_breakout_threshold": 0.15,  # 15% price movement
            "min_transactions_for_analysis": 50,  # Minimum transactions needed
            "analysis_time_window": 24,  # Hours to look back for pattern detection
            "pattern_refresh_interval": 300,  # Seconds between pattern checks
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Pattern detection state
        self.last_pattern_analysis = {}  # token_address -> timestamp of last analysis
        self.detected_patterns = {}  # token_address -> list of detected patterns
        
        logger.info(f"Pattern Analysis Agent {self.id} initialized")
    
    async def _initialize(self) -> None:
        """Initialize the agent."""
        # No special initialization needed beyond the base class
        logger.info(f"Pattern Analysis Agent {self.id} ready")
    
    async def _handle_message(self, message: Message) -> None:
        """
        Handle incoming messages.
        
        Args:
            message: The message to handle.
        """
        logger.debug(f"Pattern Analysis Agent {self.id} received message: {message.type.value}")
        
        if message.type == MessageType.EVENT:
            await self._handle_event(message)
        elif message.type == MessageType.COMMAND:
            await self._handle_command(message)
        elif message.type == MessageType.QUERY:
            await self._handle_query(message)
        else:
            logger.warning(f"Unsupported message type: {message.type.value}")
    
    async def _handle_event(self, message: Message) -> None:
        """
        Handle event messages.
        
        Args:
            message: The event message.
        """
        event_type = message.content.get("event_type")
        
        if event_type == "new_token_discovered":
            # Handle a new token event
            token_address = message.content.get("token_address")
            
            if token_address:
                logger.info(f"Processing new token for pattern analysis: {token_address}")
                # Schedule pattern analysis for this token
                await self._analyze_token_patterns(token_address)
        
        elif event_type == "token_updated":
            # Handle token data update event
            token_address = message.content.get("token_address")
            
            if token_address:
                logger.info(f"Token updated, checking patterns: {token_address}")
                # Re-analyze patterns with updated data
                await self._analyze_token_patterns(token_address)
    
    async def _handle_command(self, message: Message) -> None:
        """
        Handle command messages.
        
        Args:
            message: The command message.
        """
        command = message.content.get("command")
        
        if command == "analyze_token":
            # Handle command to analyze patterns for a specific token
            token_address = message.content.get("token_address")
            force_refresh = message.content.get("force_refresh", False)
            
            if token_address:
                try:
                    # Analyze token patterns
                    patterns = await self._analyze_token_patterns(token_address, force_refresh)
                    
                    # Send response with detected patterns
                    response = Message(
                        msg_type=MessageType.RESPONSE,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "token_address": token_address,
                            "patterns": patterns
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(response)
                except Exception as e:
                    error_msg = Message(
                        msg_type=MessageType.ERROR,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "error": f"Failed to analyze token patterns: {str(e)}",
                            "token_address": token_address
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(error_msg)
        
        elif command == "set_knowledge_base":
            # Set the knowledge base
            kb_reference = message.content.get("knowledge_base")
            if kb_reference:
                self.knowledge_base = kb_reference
                logger.info(f"Set knowledge base for {self.id}")
                
                # Send acknowledgement
                response = Message(
                    msg_type=MessageType.RESPONSE,
                    sender_id=self.id,
                    target_id=message.sender_id,
                    content={"status": "success"},
                    correlation_id=message.correlation_id
                )
                await self.send_message(response)
            else:
                error_msg = Message(
                    msg_type=MessageType.ERROR,
                    sender_id=self.id,
                    target_id=message.sender_id,
                    content={"error": "No knowledge base provided"},
                    correlation_id=message.correlation_id
                )
                await self.send_message(error_msg)
        
        elif command == "update_config":
            # Update configuration parameters
            new_config = message.content.get("config", {})
            if new_config:
                self.config.update(new_config)
                logger.info(f"Updated configuration for {self.id}")
                
                # Send acknowledgement
                response = Message(
                    msg_type=MessageType.RESPONSE,
                    sender_id=self.id,
                    target_id=message.sender_id,
                    content={
                        "status": "success",
                        "updated_config": self.config
                    },
                    correlation_id=message.correlation_id
                )
                await self.send_message(response)
            else:
                error_msg = Message(
                    msg_type=MessageType.ERROR,
                    sender_id=self.id,
                    target_id=message.sender_id,
                    content={"error": "No configuration provided"},
                    correlation_id=message.correlation_id
                )
                await self.send_message(error_msg)
        
        else:
            logger.warning(f"Unknown command: {command}")
            # Send error response
            error_msg = Message(
                msg_type=MessageType.ERROR,
                sender_id=self.id,
                target_id=message.sender_id,
                content={"error": f"Unknown command: {command}"},
                correlation_id=message.correlation_id
            )
            await self.send_message(error_msg)
    
    async def _handle_query(self, message: Message) -> None:
        """
        Handle query messages.
        
        Args:
            message: The query message.
        """
        query_type = message.content.get("query_type")
        
        if query_type == "get_token_patterns":
            # Handle query for patterns of a specific token
            token_address = message.content.get("token_address")
            
            if token_address:
                # Get patterns from cache or analyze if needed
                if token_address in self.detected_patterns:
                    patterns = self.detected_patterns[token_address]
                else:
                    patterns = await self._analyze_token_patterns(token_address)
                
                # Send response with patterns
                response = Message(
                    msg_type=MessageType.RESPONSE,
                    sender_id=self.id,
                    target_id=message.sender_id,
                    content={
                        "token_address": token_address,
                        "patterns": patterns
                    },
                    correlation_id=message.correlation_id
                )
                await self.send_message(response)
        
        elif query_type == "get_promising_tokens":
            # Handle query for tokens with high-confidence patterns
            count = message.content.get("count", 10)
            pattern_types = message.content.get("pattern_types", None)
            min_severity = message.content.get("min_severity", PatternSeverity.MEDIUM)
            
            # Get promising tokens
            promising_tokens = await self._get_promising_tokens(
                count=count,
                pattern_types=pattern_types,
                min_severity=min_severity
            )
            
            # Send response
            response = Message(
                msg_type=MessageType.RESPONSE,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "promising_tokens": promising_tokens
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
        
        else:
            logger.warning(f"Unknown query type: {query_type}")
            # Send error response
            error_msg = Message(
                msg_type=MessageType.ERROR,
                sender_id=self.id,
                target_id=message.sender_id,
                content={"error": f"Unknown query type: {query_type}"},
                correlation_id=message.correlation_id
            )
            await self.send_message(error_msg)
    
    async def _analyze_token_patterns(
        self,
        token_address: str,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Analyze patterns for a specific token.
        
        Args:
            token_address: The token's address.
            force_refresh: Whether to force a pattern re-analysis.
            
        Returns:
            List of detected patterns.
        """
        # Check if we have recently analyzed this token
        current_time = time.time()
        if (not force_refresh and 
            token_address in self.last_pattern_analysis and
            current_time - self.last_pattern_analysis[token_address] < self.config["pattern_refresh_interval"]):
            # Return cached patterns
            return self.detected_patterns.get(token_address, [])
        
        # We need to get token data from the knowledge base
        if self.knowledge_base is None:
            logger.error("Cannot analyze token patterns: no knowledge base available")
            return []
        
        try:
            # Get token data from knowledge base
            entry = await self.knowledge_base.get_entry(f"token:{token_address}")
            if not entry:
                logger.warning(f"No token data found for {token_address}")
                return []
            
            token_data = entry.data
            
            # In a real implementation, we would also get transaction history
            # For this example, we'll use placeholder data in the token entry
            
            # List to store detected patterns
            patterns = []
            
            # Analyze various patterns
            
            # 1. Volume spike detection
            volume_spike = await self._detect_volume_spike(token_data)
            if volume_spike:
                patterns.append(volume_spike)
            
            # 2. Buy pressure detection
            buy_pressure = await self._detect_buy_pressure(token_data)
            if buy_pressure:
                patterns.append(buy_pressure)
            
            # 3. Whale entry detection
            whale_entry = await self._detect_whale_entry(token_data)
            if whale_entry:
                patterns.append(whale_entry)
            
            # 4. Holder growth detection
            holder_growth = await self._detect_holder_growth(token_data)
            if holder_growth:
                patterns.append(holder_growth)
            
            # 5. Price breakout detection
            price_breakout = await self._detect_price_breakout(token_data)
            if price_breakout:
                patterns.append(price_breakout)
            
            # Store results
            self.detected_patterns[token_address] = patterns
            self.last_pattern_analysis[token_address] = current_time
            
            # If any significant patterns were detected, store in knowledge base
            if any(p["severity"] in [PatternSeverity.HIGH, PatternSeverity.CRITICAL] for p in patterns):
                await self.knowledge_base.add_entry(
                    entry_id=f"pattern:{token_address}",
                    entry_type=EntryType.PATTERN,
                    data={
                        "token_address": token_address,
                        "patterns": patterns,
                        "analysis_time": datetime.utcnow().isoformat()
                    },
                    metadata={
                        "high_confidence": any(p["severity"] == PatternSeverity.CRITICAL for p in patterns)
                    },
                    source_agent_id=self.id,
                    tags=["pattern", "analysis"]
                )
                
                # Notify other agents about significant patterns
                if any(p["severity"] == PatternSeverity.CRITICAL for p in patterns):
                    event_msg = Message(
                        msg_type=MessageType.EVENT,
                        sender_id=self.id,
                        content={
                            "event_type": "critical_pattern_detected",
                            "token_address": token_address,
                            "patterns": [p for p in patterns if p["severity"] == PatternSeverity.CRITICAL],
                            "detection_time": datetime.utcnow().isoformat()
                        }
                    )
                    await self.send_message(event_msg)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing token patterns for {token_address}: {str(e)}")
            return []
    
    async def _detect_volume_spike(self, token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect volume spikes in token trading.
        
        Args:
            token_data: Token data including market info.
            
        Returns:
            Pattern data if a volume spike is detected, None otherwise.
        """
        try:
            # In a real implementation, we would analyze historical volume data
            # For this example, we'll use placeholder logic
            
            # Extract volume data
            current_volume = token_data.get("market", {}).get("volume_24h", 0)
            
            # Placeholder for average volume
            # In a real implementation, this would come from historical data
            avg_volume = current_volume / 2.5  # Just for example
            
            # Check for volume spike
            if current_volume > avg_volume * self.config["volume_spike_threshold"]:
                severity = PatternSeverity.MEDIUM
                
                # Higher severity for more extreme spikes
                if current_volume > avg_volume * 5:
                    severity = PatternSeverity.HIGH
                if current_volume > avg_volume * 10:
                    severity = PatternSeverity.CRITICAL
                
                return {
                    "type": PatternType.VOLUME_SPIKE,
                    "severity": severity,
                    "details": {
                        "current_volume": current_volume,
                        "average_volume": avg_volume,
                        "spike_ratio": current_volume / avg_volume
                    },
                    "detection_time": datetime.utcnow().isoformat()
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Error detecting volume spike: {str(e)}")
            return None
    
    async def _detect_buy_pressure(self, token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect strong buy pressure.
        
        Args:
            token_data: Token data including transaction info.
            
        Returns:
            Pattern data if strong buy pressure is detected, None otherwise.
        """
        try:
            # In a real implementation, we would analyze buy vs. sell transactions
            # For this example, we'll use placeholder logic
            
            # Placeholder buy ratio (buys / total transactions)
            # In a real implementation, this would come from transaction data
            buy_ratio = 0.7  # Just for example
            
            # Check for buy pressure
            if buy_ratio > self.config["buy_pressure_threshold"]:
                severity = PatternSeverity.MEDIUM
                
                # Higher severity for stronger buy pressure
                if buy_ratio > 0.8:
                    severity = PatternSeverity.HIGH
                if buy_ratio > 0.9:
                    severity = PatternSeverity.CRITICAL
                
                return {
                    "type": PatternType.BUY_PRESSURE,
                    "severity": severity,
                    "details": {
                        "buy_ratio": buy_ratio,
                        "threshold": self.config["buy_pressure_threshold"]
                    },
                    "detection_time": datetime.utcnow().isoformat()
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Error detecting buy pressure: {str(e)}")
            return None
    
    async def _detect_whale_entry(self, token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect whale entry (large holder acquisition).
        
        Args:
            token_data: Token data including holder info.
            
        Returns:
            Pattern data if whale entry is detected, None otherwise.
        """
        try:
            # In a real implementation, we would track changes in holder distribution
            # For this example, we'll use placeholder logic
            
            # Extract holder data
            holders = token_data.get("holders", {}).get("distribution", [])
            
            # Check for whales (holders with significant percentage)
            whales = [h for h in holders if h.get("percentage", 0) >= self.config["whale_entry_min_percentage"]]
            
            if whales:
                # In a real implementation, we'd check if these are new whales
                # For example, compare with previous holder snapshot
                
                max_whale_percentage = max([w.get("percentage", 0) for w in whales])
                
                severity = PatternSeverity.MEDIUM
                
                # Higher severity for larger whale positions
                if max_whale_percentage > 10:
                    severity = PatternSeverity.HIGH
                if max_whale_percentage > 20:
                    severity = PatternSeverity.CRITICAL
                
                return {
                    "type": PatternType.WHALE_ENTRY,
                    "severity": severity,
                    "details": {
                        "whale_count": len(whales),
                        "max_percentage": max_whale_percentage,
                        "whales": whales
                    },
                    "detection_time": datetime.utcnow().isoformat()
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Error detecting whale entry: {str(e)}")
            return None
    
    async def _detect_holder_growth(self, token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect rapid growth in token holders.
        
        Args:
            token_data: Token data including holder info.
            
        Returns:
            Pattern data if rapid holder growth is detected, None otherwise.
        """
        try:
            # In a real implementation, we would track changes in holder count over time
            # For this example, we'll use placeholder logic
            
            # Extract current holder count
            current_holders = token_data.get("holders", {}).get("count", 0)
            
            # Placeholder for previous holder count
            # In a real implementation, this would come from historical data
            previous_holders = int(current_holders * 0.85)  # Just for example
            
            # Calculate growth percentage
            if previous_holders > 0:
                growth_percentage = (current_holders - previous_holders) / previous_holders * 100
                
                # Check for significant growth
                if growth_percentage >= self.config["holder_growth_threshold"]:
                    severity = PatternSeverity.MEDIUM
                    
                    # Higher severity for more rapid growth
                    if growth_percentage >= 25:
                        severity = PatternSeverity.HIGH
                    if growth_percentage >= 50:
                        severity = PatternSeverity.CRITICAL
                    
                    return {
                        "type": PatternType.HOLDER_GROWTH,
                        "severity": severity,
                        "details": {
                            "current_holders": current_holders,
                            "previous_holders": previous_holders,
                            "growth_percentage": growth_percentage
                        },
                        "detection_time": datetime.utcnow().isoformat()
                    }
            
            return None
        
        except Exception as e:
            logger.error(f"Error detecting holder growth: {str(e)}")
            return None
    
    async def _detect_price_breakout(self, token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect price breakouts.
        
        Args:
            token_data: Token data including price info.
            
        Returns:
            Pattern data if a price breakout is detected, None otherwise.
        """
        try:
            # In a real implementation, we would analyze price movement patterns
            # For this example, we'll use placeholder logic
            
            # Extract price data
            price_change = token_data.get("market", {}).get("price_change_24h", 0)
            
            # Check for significant price movement
            if abs(price_change) >= self.config["price_breakout_threshold"] * 100:  # convert to percentage
                # Determine if it's an upward or downward breakout
                breakout_direction = "upward" if price_change > 0 else "downward"
                
                # We're primarily interested in upward breakouts
                if breakout_direction == "upward":
                    severity = PatternSeverity.MEDIUM
                    
                    # Higher severity for larger breakouts
                    if price_change >= 30:
                        severity = PatternSeverity.HIGH
                    if price_change >= 50:
                        severity = PatternSeverity.CRITICAL
                    
                    return {
                        "type": PatternType.PRICE_BREAKOUT,
                        "severity": severity,
                        "details": {
                            "price_change": price_change,
                            "direction": breakout_direction,
                            "threshold": self.config["price_breakout_threshold"] * 100
                        },
                        "detection_time": datetime.utcnow().isoformat()
                    }
            
            return None
        
        except Exception as e:
            logger.error(f"Error detecting price breakout: {str(e)}")
            return None
    
    async def _get_promising_tokens(
        self,
        count: int = 10,
        pattern_types: Optional[List[str]] = None,
        min_severity: str = PatternSeverity.MEDIUM
    ) -> List[Dict[str, Any]]:
        """
        Get tokens with promising patterns.
        
        Args:
            count: Maximum number of tokens to return.
            pattern_types: Optional list of pattern types to filter by.
            min_severity: Minimum pattern severity to consider.
            
        Returns:
            List of promising tokens with their patterns.
        """
        promising_tokens = []
        
        # In a real implementation, we would query the knowledge base
        # For this example, we'll use our cached pattern data
        
        # Convert severity to numeric value for comparison
        severity_values = {
            PatternSeverity.LOW: 1,
            PatternSeverity.MEDIUM: 2,
            PatternSeverity.HIGH: 3,
            PatternSeverity.CRITICAL: 4
        }
        min_severity_value = severity_values.get(min_severity, 2)
        
        # Score each token based on its patterns
        token_scores = []
        for token_address, patterns in self.detected_patterns.items():
            # Filter by pattern types if specified
            if pattern_types:
                relevant_patterns = [p for p in patterns if p["type"] in pattern_types]
            else:
                relevant_patterns = patterns
            
            # Filter by minimum severity
            relevant_patterns = [
                p for p in relevant_patterns 
                if severity_values.get(p["severity"], 0) >= min_severity_value
            ]
            
            if relevant_patterns:
                # Calculate a score based on pattern severity
                score = sum(severity_values.get(p["severity"], 0) for p in relevant_patterns)
                
                token_scores.append((token_address, score, relevant_patterns))
        
        # Sort by score in descending order
        token_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top tokens
        for token_address, score, patterns in token_scores[:count]:
            promising_tokens.append({
                "token_address": token_address,
                "score": score,
                "patterns": patterns
            })
        
        return promising_tokens
    
    async def _cleanup(self) -> None:
        """Clean up resources when stopping the agent."""
        # Nothing special to clean up
        logger.info(f"Pattern Analysis Agent {self.id} cleaned up")
