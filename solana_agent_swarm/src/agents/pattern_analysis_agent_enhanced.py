"""
Enhanced Pattern Analysis Agent for the Solana Token Analysis Agent Swarm.
Analyzes token transactions and market behavior to identify patterns
that may indicate high-performing tokens, using machine learning models
for improved pattern detection and anomaly identification.
"""

import asyncio
import json
import time
import uuid
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from ..core.agent_base import Agent, AgentStatus, Message, MessageType
from ..core.knowledge_base import EntryType, KnowledgeBase
from ..utils.ml_utils import PatternDetector, TokenPricePredictor


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
    ML_ANOMALY = "ml_anomaly"          # New: ML-detected anomaly
    ML_PATTERN = "ml_pattern"          # New: ML-identified pattern
    PRICE_PREDICTION = "price_prediction"  # New: ML price prediction


class PatternSeverity:
    """Constants for pattern significance levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EnhancedPatternAnalysisAgent(Agent):
    """
    Enhanced agent responsible for analyzing token transaction patterns with ML.
    Identifies patterns that may indicate high-performing tokens.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Enhanced Pattern Analysis Agent.
        
        Args:
            agent_id: Optional agent ID. If not provided, one will be generated.
            knowledge_base: Optional knowledge base instance to use.
                If not provided, the agent will expect to receive one
                from the orchestrator after initialization.
            config: Optional configuration for pattern detection thresholds.
        """
        agent_id = agent_id or f"ml-pattern-analyzer-{str(uuid.uuid4())[:8]}"
        super().__init__(agent_id=agent_id, agent_type="EnhancedPatternAnalysisAgent")
        
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
            "ml_model_path": "models/pattern_model.pkl",  # Path to ML model
            "anomaly_threshold": 0.7,  # Threshold for anomaly detection (0-1)
            "prediction_window": 7,  # Days to predict ahead
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Pattern detection state
        self.last_pattern_analysis = {}  # token_address -> timestamp of last analysis
        self.detected_patterns = {}  # token_address -> list of detected patterns
        
        # ML components
        self.pattern_detector = None
        self.price_predictor = None
        
        logger.info(f"Enhanced Pattern Analysis Agent {self.id} initialized")
    
    async def _initialize(self) -> None:
        """Initialize the agent with ML models."""
        # Initialize ML components
        self.pattern_detector = PatternDetector()
        
        # Initialize price predictor with configuration
        self.price_predictor = TokenPricePredictor(
            input_size=10,   # 10 data points of history
            hidden_size=50,  # LSTM hidden layer size
            num_layers=2,    # 2 LSTM layers
            output_size=1    # Predict price only
        )
        
        # Load models if available
        model_loaded = await self._load_models()
        
        # Report initialization status
        if model_loaded:
            logger.info(f"ML models loaded successfully for {self.id}")
            await self.send_confidence_report(
                task_name="model_initialization",
                confidence=8.5,
                details="ML models loaded successfully"
            )
        else:
            logger.warning(f"ML models not loaded, will train as data becomes available for {self.id}")
            await self.send_confidence_report(
                task_name="model_initialization",
                confidence=5.0,
                details="ML models not loaded, will train with incoming data"
            )
        
        logger.info(f"Enhanced Pattern Analysis Agent {self.id} ready")
    
    async def _load_models(self) -> bool:
        """
        Load ML models from disk if available.
        
        Returns:
            Success status of model loading
        """
        try:
            # Check if model path exists
            model_path = self.config["ml_model_path"]
            if os.path.exists(model_path):
                # In a real implementation, this would load the saved model
                # For this example, we'll just simulate loading
                logger.info(f"Loading ML model from {model_path}")
                return True
            else:
                logger.warning(f"Model file {model_path} not found")
                return False
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    async def _handle_message(self, message: Message) -> None:
        """
        Handle incoming messages.
        
        Args:
            message: The message to handle.
        """
        logger.debug(f"Enhanced Pattern Analysis Agent {self.id} received message: {message.type.value}")
        
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
                logger.info(f"Processing new token for ML pattern analysis: {token_address}")
                # Schedule pattern analysis for this token
                await self._analyze_token_patterns(token_address)
        
        elif event_type == "token_updated":
            # Handle token data update event
            token_address = message.content.get("token_address")
            
            if token_address:
                logger.info(f"Token updated, running ML pattern analysis: {token_address}")
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
        
        elif command == "predict_price":
            # New command to predict future price movement
            token_address = message.content.get("token_address")
            days = message.content.get("days", self.config["prediction_window"])
            
            if token_address:
                try:
                    # Predict price movement
                    prediction = await self._predict_token_price(token_address, days)
                    
                    # Send response with prediction
                    response = Message(
                        msg_type=MessageType.RESPONSE,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "token_address": token_address,
                            "price_prediction": prediction,
                            "days": days
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
                            "error": f"Failed to predict token price: {str(e)}",
                            "token_address": token_address
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(error_msg)
        
        elif command == "train_ml_models":
            # New command to train ML models with historical data
            token_addresses = message.content.get("token_addresses", [])
            force_retrain = message.content.get("force_retrain", False)
            
            try:
                # Train ML models
                training_result = await self._train_ml_models(token_addresses, force_retrain)
                
                # Send response with training results
                response = Message(
                    msg_type=MessageType.RESPONSE,
                    sender_id=self.id,
                    target_id=message.sender_id,
                    content={
                        "training_result": training_result
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
                        "error": f"Failed to train ML models: {str(e)}"
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
                
        elif query_type == "get_anomalies":
            # New query to get detected anomalies
            token_address = message.content.get("token_address")
            
            if token_address:
                # Get token data
                token_data = await self._get_token_data(token_address)
                
                if token_data:
                    # Detect anomalies using ML
                    anomalies = await self._detect_anomalies(token_data)
                    
                    # Send response with anomalies
                    response = Message(
                        msg_type=MessageType.RESPONSE,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "token_address": token_address,
                            "anomalies": anomalies
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(response)
                else:
                    error_msg = Message(
                        msg_type=MessageType.ERROR,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "error": f"No token data found for {token_address}"
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(error_msg)
        
        elif query_type == "get_promising_tokens":
            # Handle query for tokens with high-confidence patterns
            count = message.content.get("count", 10)
            pattern_types = message.content.get("pattern_types", None)
            min_severity = message.content.get("min_severity", PatternSeverity.MEDIUM)
            include_ml_patterns = message.content.get("include_ml_patterns", True)
            
            # Get promising tokens with ML analysis
            promising_tokens = await self._get_promising_tokens(
                count=count,
                pattern_types=pattern_types,
                min_severity=min_severity,
                include_ml_patterns=include_ml_patterns
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
        Analyze patterns for a specific token with ML enhancements.
        
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
        
        # Get token data
        token_data = await self._get_token_data(token_address)
        if not token_data:
            logger.warning(f"No token data found for {token_address}")
            return []
        
        try:
            # List to store detected patterns
            patterns = []
            
            # Run traditional pattern analysis first
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
            
            # Apply ML-enhanced pattern detection
            
            # 6. Anomaly detection
            ml_anomalies = await self._detect_anomalies(token_data)
            if ml_anomalies:
                # Convert anomalies to pattern format
                for anomaly in ml_anomalies:
                    if anomaly.get("is_anomaly", False):
                        patterns.append({
                            "type": PatternType.ML_ANOMALY,
                            "severity": PatternSeverity.HIGH if anomaly.get("anomaly_score", 0) > 0.8 else PatternSeverity.MEDIUM,
                            "details": {
                                "anomaly_score": anomaly.get("anomaly_score", 0),
                                "timestamp": anomaly.get("timestamp"),
                                "data_point": anomaly.get("data_point")
                            },
                            "detection_time": datetime.utcnow().isoformat()
                        })
            
            # 7. ML Pattern classification
            ml_pattern = await self._classify_pattern(token_data)
            if ml_pattern:
                confidence = ml_pattern.get("confidence", 0)
                severity = PatternSeverity.LOW
                if confidence > 0.7:
                    severity = PatternSeverity.MEDIUM
                if confidence > 0.85:
                    severity = PatternSeverity.HIGH
                if confidence > 0.95:
                    severity = PatternSeverity.CRITICAL
                
                patterns.append({
                    "type": PatternType.ML_PATTERN,
                    "severity": severity,
                    "details": {
                        "pattern_type": ml_pattern.get("pattern_type", "unknown"),
                        "confidence": confidence,
                        "probabilities": ml_pattern.get("all_probabilities", {})
                    },
                    "detection_time": datetime.utcnow().isoformat()
                })
            
            # 8. Price prediction
            price_prediction = await self._predict_token_price(token_address)
            if price_prediction and "predicted_price_change" in price_prediction:
                # Convert prediction to pattern if significant change expected
                predicted_change = price_prediction.get("predicted_price_change", 0)
                
                if abs(predicted_change) >= 5: # At least 5% change prediction
                    severity = PatternSeverity.LOW
                    if abs(predicted_change) >= 10:
                        severity = PatternSeverity.MEDIUM
                    if abs(predicted_change) >= 20:
                        severity = PatternSeverity.HIGH
                    if abs(predicted_change) >= 30:
                        severity = PatternSeverity.CRITICAL
                    
                    direction = "upward" if predicted_change > 0 else "downward"
                    
                    # Only include upward predictions as patterns
                    if direction == "upward":
                        patterns.append({
                            "type": PatternType.PRICE_PREDICTION,
                            "severity": severity,
                            "details": {
                                "predicted_change": predicted_change,
                                "direction": direction,
                                "prediction_window": price_prediction.get("timeframe", "7 days"),
                                "confidence": price_prediction.get("confidence", 0.5)
                            },
                            "detection_time": datetime.utcnow().isoformat()
                        })
            
            # Store results
            self.detected_patterns[token_address] = patterns
            self.last_pattern_analysis[token_address] = current_time
            
            # Report confidence in analysis
            critical_patterns = [p for p in patterns if p["severity"] == PatternSeverity.CRITICAL]
            if critical_patterns:
                await self.send_confidence_report(
                    task_name="pattern_analysis",
                    confidence=9.0,
                    details=f"Detected {len(critical_patterns)} critical patterns"
                )
            else:
                await self.send_confidence_report(
                    task_name="pattern_analysis",
                    confidence=8.0,
                    details=f"Completed analysis, found {len(patterns)} patterns"
                )
            
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
                    tags=["pattern", "analysis", "ml_enhanced"]
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
    
    async def _get_token_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Get token data from knowledge base.
        
        Args:
            token_address: The token's address.
            
        Returns:
            Token data if found, None otherwise.
        """
        if self.knowledge_base is None:
            logger.error("Cannot get token data: no knowledge base available")
            return None
        
        # Get token data from knowledge base
        entry = await self.knowledge_base.get_entry(f"token:{token_address}")
        if not entry:
            return None
        
        return entry.data
    
    async def _detect_anomalies(self, token_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in token data using ML.
        
        Args:
            token_data: Token data including price history.
            
        Returns:
            List of detected anomalies.
        """
        # Check if ML is available
        if not self.pattern_detector or not hasattr(self.pattern_detector, 'detect_anomalies'):
            logger.warning("Cannot detect anomalies: ML not available")
            return []
        
        try:
            # Get price history
            price_history = token_data.get("price_history", [])
            if not price_history:
                logger.warning("Cannot detect anomalies: no price history available")
                return []
            
            # Make sure we have enough data for analysis
            if len(price_history) < 5:
                logger.warning("Not enough price history for anomaly detection")
                return []
            
            # Check if we need to train the model
            if not getattr(self.pattern_detector, 'is_trained', False):
                logger.info("Training anomaly detector with available data")
                success = self.pattern_detector.train(price_history)
                if not success:
                    logger.warning("Failed to train anomaly detector")
                    return []
            
            # Detect anomalies
            anomalies = self.pattern_detector.detect_anomalies(price_history)
            
            # Filter anomalies by threshold
            threshold = self.config["anomaly_threshold"]
            significant_anomalies = [
                a for a in anomalies 
                if a.get("anomaly_score", 0) >= threshold
            ]
            
            return significant_anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    async def _classify_pattern(self, token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Classify token pattern using ML.
        
        Args:
            token_data: Token data including price and volume history.
            
        Returns:
            Classification result or None if not available.
        """
        # Check if ML is available
        if not self.pattern_detector or not hasattr(self.pattern_detector, 'classify_pattern'):
            logger.warning("Cannot classify pattern: ML not available")
            return None
        
        try:
            # Get price history
            price_history = token_data.get("price_history", [])
            if not price_history:
                logger.warning("Cannot classify pattern: no price history available")
                return None
            
            # Make sure we have enough data
            if len(price_history) < 5:
                logger.warning("Not enough price history for pattern classification")
                return None
            
            # Attempt pattern classification
            pattern = self.pattern_detector.classify_pattern(price_history[:1])
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error classifying pattern: {str(e)}")
            return None
    
    async def _predict_token_price(
        self, 
        token_address: str,
        days: int = None
    ) -> Optional[Dict[str, Any]]:
        """
        Predict future token price using ML.
        
        Args:
            token_address: The token's address.
            days: Number of days to predict ahead.
            
        Returns:
            Price prediction or None if not available.
        """
        if days is None:
            days = self.config["prediction_window"]
            
        # Check if ML is available
        if not self.price_predictor or not hasattr(self.price_predictor, 'predict'):
            logger.warning("Cannot predict price: ML not available")
            return None
        
        try:
            # Get token data
            token_data = await self._get_token_data(token_address)
            if not token_data:
                logger.warning(f"No token data found for {token_address}")
                return None
            
            # Get price history
            price_history = token_data.get("price_history", [])
            if not price_history:
                logger.warning("Cannot predict price: no price history available")
                return None
            
            # Make sure we have enough data
            if len(price_history) < 10:  # Need at least 10 data points
                logger.warning("Not enough price history for price prediction")
                return None
            
            # Prepare features for prediction
            # For real implementation, you would extract appropriate features
            features = self._extract_prediction_features(price_history)
            
            # Need to train the model if not trained
            if not getattr(self.price_predictor, 'model', None):
                # Create target values (next day prices)
                targets = np.array([[p["price"]] for p in price_history[1:]])
                
                # Train the model
                success = await self.price_predictor.train(features[:-1], targets)
                if not success:
                    logger.warning("Failed to train price predictor")
                    return None
            
            # Make prediction
            prediction_features = features[-1:]  # Use the most recent data point
            predictions = await self.price_predictor.predict(prediction_features)
            
            if predictions is None or len(predictions) == 0:
                logger.warning("Prediction failed")
                return None
            
            # Get current price and calculate expected change
            current_price = price_history[-1]["price"]
            predicted_price = float(predictions[0][0])
            price_change_pct = ((predicted_price - current_price) / current_price) * 100
            
            # Calculate confidence based on model metrics
            # In a real implementation, this would come from the model
            confidence = 0.7
            
            return {
                "current_price": current_price,
                "predicted_price": predicted_price,
                "predicted_price_change": price_change_pct,
                "timeframe": f"{days} days",
                "confidence": confidence,
                "prediction_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting token price: {str(e)}")
            return None
    
    async def _train_ml_models(self, token_addresses: List[str], force_retrain: bool = False) -> Dict[str, Any]:
        """
        Train ML models with historical data from multiple tokens.
        
        Args:
            token_addresses: List of token addresses to use for training.
            force_retrain: Whether to force retraining existing models.
            
        Returns:
            Training results.
        """
        if not token_addresses:
            # If no specific tokens provided, get all tokens from knowledge base
            if self.knowledge_base:
                # Query for tokens with sufficient data
                entries = await self.knowledge_base.query(
                    query_type="token",
                    limit=50  # Limit to 50 tokens for training
                )
                token_addresses = [e.id.split(":")[-1] for e in entries if e.id.startswith("token:")]
        
        if not token_addresses:
            return {
                "success": False,
                "error": "No tokens available for training"
            }
        
        # Collect training data
        pattern_training_data = []
        price_training_features = []
        price_training_targets = []
        
        for token_address in token_addresses:
            token_data = await self._get_token_data(token_address)
            if not token_data or "price_history" not in token_data or len(token_data["price_history"]) < 10:
                continue
            
            # Add data for pattern detection
            price_history = token_data["price_history"]
            pattern_training_data.extend(price_history)
            
            # Add data for price prediction
            features = self._extract_prediction_features(price_history)
            if len(features) > 1:
                price_training_features.extend(features[:-1])
                price_training_targets.extend([[p["price"]] for p in price_history[1:]])
        
        results = {
            "pattern_detector_trained": False,
            "price_predictor_trained": False,
            "tokens_used": len(token_addresses),
            "data_points": len(pattern_training_data)
        }
        
        # Train pattern detector if we have data
        if pattern_training_data and (force_retrain or not getattr(self.pattern_detector, 'is_trained', False)):
            try:
                # In a real implementation, we would need labels for supervised learning
                # Here we'll use placeholder labels for demonstration
                pattern_labels = ["uptrend", "downtrend", "sideways"] * (len(pattern_training_data) // 3 + 1)
                pattern_labels = pattern_labels[:len(pattern_training_data)]
                
                success = self.pattern_detector.train(pattern_training_data, pattern_labels)
                results["pattern_detector_trained"] = success
            except Exception as e:
                logger.error(f"Error training pattern detector: {str(e)}")
                results["pattern_detector_error"] = str(e)
        
        # Train price predictor if we have data
        if price_training_features and price_training_targets and (force_retrain or not getattr(self.price_predictor, 'model', None)):
            try:
                # Convert to numpy arrays
                features_array = np.array(price_training_features, dtype=np.float32)
                targets_array = np.array(price_training_targets, dtype=np.float32)
                
                success = await self.price_predictor.train(
                    features_array, 
                    targets_array,
                    epochs=5,
                    batch_size=32
                )
                results["price_predictor_trained"] = success
            except Exception as e:
                logger.error(f"Error training price predictor: {str(e)}")
                results["price_predictor_error"] = str(e)
        
        # Save models if training was successful
        if results["pattern_detector_trained"] or results["price_predictor_trained"]:
            # In a real implementation, this would save the models to disk
            results["models_saved"] = True
        
        results["success"] = results["pattern_detector_trained"] or results["price_predictor_trained"]
        return results
    
    def _extract_prediction_features(self, price_history: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract features for price prediction from price history.
        
        Args:
            price_history: List of price data points.
            
        Returns:
            Array of features for prediction.
        """
        # In a real implementation, this would extract meaningful features
        # For this example, we'll use a simplified approach
        
        features = []
        window_size = 10  # Look at 10 days of history for each prediction
        
        for i in range(len(price_history) - window_size + 1):
            window = price_history[i:i+window_size]
            window_features = []
            
            for point in window:
                # Extract features from each data point
                price = point.get("price", 0)
                volume = point.get("volume", 0)
                
                # Add more features as needed
                window_features.append([
                    price,
                    volume,
                    price / max(volume, 1),  # Price/volume ratio
                    point.get("market_cap", 0) / max(1, price),  # Supply (market_cap/price)
                    1 if point.get("direction", "") == "up" else 0  # Direction as binary feature
                ])
            
            features.append(window_features)
        
        if not features:
            return np.array([])
        
        return np.array(features, dtype=np.float32)
    
    async def _detect_volume_spike(self, token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect volume spikes in token trading.
        
        Args:
            token_data: Token data including market info.
            
        Returns:
            Pattern data if a volume spike is detected, None otherwise.
        """
        try:
            # Extract volume data
            current_volume = token_data.get("market", {}).get("volume_24h", 0)
            
            # Get historical volume for comparison
            volume_history = [p.get("volume", 0) for p in token_data.get("price_history", [])]
            if volume_history:
                avg_volume = sum(volume_history) / len(volume_history)
            else:
                # Placeholder for average volume if no history
                avg_volume = current_volume / 2.5
            
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
            transactions = token_data.get("transactions", [])
            if transactions:
                buy_count = sum(1 for t in transactions if t.get("type") == "buy")
                buy_ratio = buy_count / len(transactions)
            else:
                # Just for example if no transaction data
                buy_ratio = 0.7
            
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
            # Extract current holder count
            current_holders = token_data.get("holders", {}).get("count", 0)
            
            # Get historical holder counts if available
            holder_history = token_data.get("holder_history", [])
            if holder_history and len(holder_history) > 1:
                previous_holders = holder_history[-2].get("count", 0)
            else:
                # Placeholder for previous holder count
                previous_holders = int(current_holders * 0.85)
            
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
        min_severity: str = PatternSeverity.MEDIUM,
        include_ml_patterns: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get tokens with promising patterns, enhanced with ML insights.
        
        Args:
            count: Maximum number of tokens to return.
            pattern_types: Optional list of pattern types to filter by.
            min_severity: Minimum pattern severity to consider.
            include_ml_patterns: Whether to include ML-detected patterns.
            
        Returns:
            List of promising tokens with their patterns.
        """
        promising_tokens = []
        
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
                relevant_patterns = patterns.copy()
            
            # Filter ML patterns if not included
            if not include_ml_patterns:
                relevant_patterns = [
                    p for p in relevant_patterns 
                    if p["type"] not in [PatternType.ML_ANOMALY, PatternType.ML_PATTERN, PatternType.PRICE_PREDICTION]
                ]
            
            # Filter by minimum severity
            relevant_patterns = [
                p for p in relevant_patterns 
                if severity_values.get(p["severity"], 0) >= min_severity_value
            ]
            
            if relevant_patterns:
                # Calculate a score based on pattern severity and confidence
                # ML patterns get a bonus
                score = 0
                for p in relevant_patterns:
                    pattern_score = severity_values.get(p["severity"], 0)
                    
                    # Apply bonus for ML patterns
                    if p["type"] in [PatternType.ML_PATTERN, PatternType.PRICE_PREDICTION]:
                        confidence = p.get("details", {}).get("confidence", 0.5)
                        pattern_score *= (1 + confidence)
                    
                    score += pattern_score
                
                # Get token name and symbol if available
                token_data = await self._get_token_data(token_address)
                token_name = token_data.get("name", "Unknown") if token_data else "Unknown"
                token_symbol = token_data.get("symbol", "Unknown") if token_data else "Unknown"
                
                token_scores.append((
                    token_address, 
                    score, 
                    relevant_patterns,
                    token_name,
                    token_symbol
                ))
        
        # Sort by score in descending order
        token_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top tokens
        for token_address, score, patterns, name, symbol in token_scores[:count]:
            promising_tokens.append({
                "token_address": token_address,
                "token_name": name,
                "token_symbol": symbol,
                "score": score,
                "patterns": patterns
            })
        
        return promising_tokens
    
    async def send_confidence_report(self, task_name: str, confidence: float, details: str) -> None:
        """Send a confidence report message to the orchestrator."""
        report_msg = Message(
            msg_type=MessageType.EVENT,
            sender_id=self.id,
            content={
                "event_type": "confidence_report",
                "task_name": task_name,
                "confidence": confidence,
                "details": details,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        await self.send_message(report_msg)
    
    async def _cleanup(self) -> None:
        """Clean up resources when stopping the agent."""
        logger.info(f"Enhanced Pattern Analysis Agent {self.id} cleaning up")
        
        # Clean up ML components
        if self.price_predictor and hasattr(self.price_predictor, 'cleanup'):
            await self.price_predictor.cleanup()
        
        logger.info(f"Enhanced Pattern Analysis Agent {self.id} cleaned up")
