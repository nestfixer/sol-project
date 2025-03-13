"""
Enhanced Risk Assessment Agent for the Solana Token Analysis Agent Swarm.
Evaluates token risks, contract safety, developer history, and
liquidity to identify potential scams and risky investments.
Features enhanced contract vulnerability detection, cross-chain risk analysis,
Switchboard oracle data integration, and comprehensive liquidity health scoring.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from loguru import logger

from ..core.agent_base import Agent, AgentStatus, Message, MessageType
from ..core.knowledge_base import EntryType, KnowledgeBase
from ..utils.solana_ecosystem import HeliusClient, JupiterClient, SPLTokenChecker
from ..utils.ml_utils import PatternDetector


class RiskFactorType:
    """Constants for different risk factor types."""
    CONTRACT_RISK = "contract_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    OWNER_CONCENTRATION = "owner_concentration"
    DEV_REPUTATION = "dev_reputation"
    TOKEN_AGE = "token_age"
    TRANSACTION_PATTERNS = "transaction_patterns"
    SOCIAL_SIGNALS = "social_signals"
    PROGRAM_VULNERABILITY = "program_vulnerability"
    CROSS_CHAIN_RISK = "cross_chain_risk"
    LIQUIDITY_FRAGILITY = "liquidity_fragility"
    TREASURY_ACTIVITY = "treasury_activity"
    ORACLE_PRICE_DEVIATION = "oracle_price_deviation"  # New: deviation between market and oracle price
    SPL_COMPLIANCE = "spl_compliance"  # New: SPL token standard compliance
    HELIUS_METADATA = "helius_metadata"  # New: Helius metadata verification


class RiskLevel:
    """Constants for risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EnhancedRiskAssessmentAgent(Agent):
    """
    Enhanced agent responsible for evaluating token risks with Solana-specific tools.
    Identifies potential scams, rug pulls, and investment risks.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Enhanced Risk Assessment Agent.
        
        Args:
            agent_id: Optional agent ID. If not provided, one will be generated.
            knowledge_base: Optional knowledge base instance to use.
                If not provided, the agent will expect to receive one
                from the orchestrator after initialization.
            config: Optional configuration for risk assessment thresholds.
        """
        agent_id = agent_id or f"enhanced-risk-assessor-{str(uuid.uuid4())[:8]}"
        super().__init__(agent_id=agent_id, agent_type="EnhancedRiskAssessmentAgent")
        
        self.knowledge_base = knowledge_base
        
        # Default configuration - can be overridden by provided config
        self.config = {
            "min_liquidity_usd": 10000.0,  # Minimum liquidity in USD
            "high_concentration_threshold": 0.3,  # 30% ownership by single wallet
            "critical_concentration_threshold": 0.5,  # 50% ownership by single wallet
            "min_holders_count": 50,  # Minimum number of holders
            "dev_reputation_weight": 0.2,  # Weight for developer reputation
            "contract_risk_weight": 0.3,  # Weight for contract risk
            "liquidity_risk_weight": 0.25,  # Weight for liquidity risk
            "concentration_risk_weight": 0.25,  # Weight for owner concentration
            "blacklisted_tokens": [],  # Known scam tokens
            "blacklisted_developers": [],  # Known scam developers
            "risk_assessment_refresh_interval": 300,  # Seconds between risk checks
            "helius_api_key": "",  # API key for Helius (to be provided in real implementation)
            "solana_rpc_url": "https://api.mainnet-beta.solana.com",  # Default RPC URL
            "oracle_price_deviation_threshold": 5.0,  # Maximum allowed % difference between oracle and market price
            "jupiter_dex_count_threshold": 2,  # Minimum number of DEXs for good liquidity distribution
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Risk assessment state
        self.last_risk_assessment = {}  # token_address -> timestamp of last assessment
        self.risk_assessments = {}  # token_address -> risk assessment result
        
        # Solana ecosystem clients
        self.helius_client = None
        self.jupiter_client = None
        self.spl_checker = None
        
        # ML pattern detector for anomaly detection
        self.pattern_detector = None
        
        logger.info(f"Enhanced Risk Assessment Agent {self.id} initialized")
    
    async def _initialize(self) -> None:
        """Initialize the agent with Solana ecosystem clients."""
        try:
            # Initialize Helius client if API key is provided
            if self.config.get("helius_api_key"):
                self.helius_client = HeliusClient(api_key=self.config["helius_api_key"])
                await self.helius_client.initialize()
                logger.info(f"Helius client initialized for {self.id}")
            else:
                logger.warning(f"No Helius API key provided, Helius features will be unavailable for {self.id}")
            
            # Initialize Jupiter client
            self.jupiter_client = JupiterClient()
            await self.jupiter_client.initialize()
            logger.info(f"Jupiter client initialized for {self.id}")
            
            # Initialize SPL token checker
            self.spl_checker = SPLTokenChecker(rpc_url=self.config["solana_rpc_url"])
            await self.spl_checker.initialize()
            logger.info(f"SPL token checker initialized for {self.id}")
            
            # Initialize pattern detector for anomaly detection
            self.pattern_detector = PatternDetector()
            logger.info(f"Pattern detector initialized for {self.id}")
            
            logger.info(f"Enhanced Risk Assessment Agent {self.id} ready")
            
            # Report initialization status and confidence
            await self.send_confidence_report(
                task_name="initialization",
                confidence=8.5,
                details="Successfully initialized with Solana ecosystem integrations"
            )
            
        except Exception as e:
            logger.error(f"Error initializing Enhanced Risk Assessment Agent: {str(e)}")
            # Report low confidence
            await self.send_confidence_report(
                task_name="initialization",
                confidence=3.0,
                details=f"Initialization failed: {str(e)}"
            )
    
    async def _handle_message(self, message: Message) -> None:
        """
        Handle incoming messages.
        
        Args:
            message: The message to handle.
        """
        logger.debug(f"Enhanced Risk Assessment Agent {self.id} received message: {message.type.value}")
        
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
                logger.info(f"Assessing risk for newly discovered token: {token_address}")
                # Schedule risk assessment for this token
                await self._assess_token_risk(token_address)
        
        elif event_type == "token_updated":
            # Handle token data update event
            token_address = message.content.get("token_address")
            
            if token_address:
                logger.info(f"Token updated, reassessing risk: {token_address}")
                # Re-assess risk with updated data
                await self._assess_token_risk(token_address)
        
        elif event_type == "critical_pattern_detected":
            # Handle critical pattern event from pattern analysis agent
            token_address = message.content.get("token_address")
            patterns = message.content.get("patterns", [])
            
            if token_address:
                logger.info(f"Critical pattern detected for {token_address}, reassessing risk")
                
                # Use the pattern information to inform risk assessment
                # In a real implementation, this would be used to update risk factors
                
                # Re-assess risk with the critical pattern information
                await self._assess_token_risk(token_address, force_refresh=True)
    
    async def _handle_command(self, message: Message) -> None:
        """
        Handle command messages.
        
        Args:
            message: The command message.
        """
        command = message.content.get("command")
        
        if command == "assess_token_risk":
            # Handle command to assess risk for a specific token
            token_address = message.content.get("token_address")
            force_refresh = message.content.get("force_refresh", False)
            
            if token_address:
                try:
                    # Assess token risk
                    risk_assessment = await self._assess_token_risk(token_address, force_refresh)
                    
                    # Send response with risk assessment
                    response = Message(
                        msg_type=MessageType.RESPONSE,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "token_address": token_address,
                            "risk_assessment": risk_assessment
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
                            "error": f"Failed to assess token risk: {str(e)}",
                            "token_address": token_address
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(error_msg)
        
        elif command == "check_spl_compliance":
            # New command to check SPL token compliance
            token_address = message.content.get("token_address")
            
            if token_address:
                try:
                    # Check SPL compliance
                    if self.spl_checker:
                        compliance = await self.spl_checker.check_token_compliance(token_address)
                        
                        # Send response with compliance check
                        response = Message(
                            msg_type=MessageType.RESPONSE,
                            sender_id=self.id,
                            target_id=message.sender_id,
                            content={
                                "token_address": token_address,
                                "compliance": compliance
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
                                "error": "SPL token checker not initialized",
                                "token_address": token_address
                            },
                            correlation_id=message.correlation_id
                        )
                        await self.send_message(error_msg)
                except Exception as e:
                    error_msg = Message(
                        msg_type=MessageType.ERROR,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "error": f"Failed to check SPL compliance: {str(e)}",
                            "token_address": token_address
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(error_msg)
        
        elif command == "analyze_liquidity":
            # New command to analyze liquidity depth
            token_address = message.content.get("token_address")
            
            if token_address:
                try:
                    # Analyze liquidity
                    if self.jupiter_client:
                        liquidity_analysis = await self.jupiter_client.analyze_liquidity_depth(token_address)
                        
                        # Send response with liquidity analysis
                        response = Message(
                            msg_type=MessageType.RESPONSE,
                            sender_id=self.id,
                            target_id=message.sender_id,
                            content={
                                "token_address": token_address,
                                "liquidity_analysis": liquidity_analysis
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
                                "error": "Jupiter client not initialized",
                                "token_address": token_address
                            },
                            correlation_id=message.correlation_id
                        )
                        await self.send_message(error_msg)
                except Exception as e:
                    error_msg = Message(
                        msg_type=MessageType.ERROR,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "error": f"Failed to analyze liquidity: {str(e)}",
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
                # Check if Helius API key is being updated
                if "helius_api_key" in new_config and new_config["helius_api_key"] != self.config.get("helius_api_key"):
                    # Re-initialize Helius client with new API key
                    try:
                        if self.helius_client:
                            # Clean up existing client
                            await self.helius_client.cleanup()
                        
                        self.helius_client = HeliusClient(api_key=new_config["helius_api_key"])
                        await self.helius_client.initialize()
                        logger.info(f"Helius client re-initialized with new API key for {self.id}")
                    except Exception as e:
                        logger.error(f"Error re-initializing Helius client: {str(e)}")
                
                # Update config
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
        
        elif command == "update_blacklists":
            # Update token and developer blacklists
            blacklisted_tokens = message.content.get("blacklisted_tokens")
            blacklisted_developers = message.content.get("blacklisted_developers")
            
            if blacklisted_tokens is not None:
                self.config["blacklisted_tokens"] = blacklisted_tokens
            
            if blacklisted_developers is not None:
                self.config["blacklisted_developers"] = blacklisted_developers
            
            logger.info(f"Updated blacklists for {self.id}")
            
            # Send acknowledgement
            response = Message(
                msg_type=MessageType.RESPONSE,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "status": "success",
                    "blacklisted_tokens": len(self.config["blacklisted_tokens"]),
                    "blacklisted_developers": len(self.config["blacklisted_developers"])
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
        
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
        
        if query_type == "get_token_risk":
            # Handle query for risk assessment of a specific token
            token_address = message.content.get("token_address")
            
            if token_address:
                # Get risk assessment from cache or assess if needed
                if token_address in self.risk_assessments:
                    risk_assessment = self.risk_assessments[token_address]
                else:
                    risk_assessment = await self._assess_token_risk(token_address)
                
                # Send response with risk assessment
                response = Message(
                    msg_type=MessageType.RESPONSE,
                    sender_id=self.id,
                    target_id=message.sender_id,
                    content={
                        "token_address": token_address,
                        "risk_assessment": risk_assessment
                    },
                    correlation_id=message.correlation_id
                )
                await self.send_message(response)
        
        elif query_type == "get_token_metadata":
            # New query for Helius token metadata
            token_address = message.content.get("token_address")
            
            if token_address:
                if self.helius_client:
                    try:
                        # Get token metadata
                        metadata = await self.helius_client.get_token_metadata(token_address)
                        
                        # Send response with metadata
                        response = Message(
                            msg_type=MessageType.RESPONSE,
                            sender_id=self.id,
                            target_id=message.sender_id,
                            content={
                                "token_address": token_address,
                                "metadata": metadata
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
                                "error": f"Failed to get token metadata: {str(e)}",
                                "token_address": token_address
                            },
                            correlation_id=message.correlation_id
                        )
                        await self.send_message(error_msg)
                else:
                    error_msg = Message(
                        msg_type=MessageType.ERROR,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "error": "Helius client not initialized",
                            "token_address": token_address
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(error_msg)
        
        elif query_type == "get_lowest_risk_tokens":
            # Handle query for tokens with lowest risk
            count = message.content.get("count", 10)
            
            # Get lowest risk tokens
            low_risk_tokens = await self._get_lowest_risk_tokens(count)
            
            # Send response
            response = Message(
                msg_type=MessageType.RESPONSE,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "low_risk_tokens": low_risk_tokens
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
    
    async def _assess_token_risk(
        self,
        token_address: str,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Assess risk for a specific token with enhanced Solana-specific analysis.
        
        Args:
            token_address: The token's address.
            force_refresh: Whether to force a risk re-assessment.
            
        Returns:
            Comprehensive risk assessment result with detailed Solana-specific analysis.
        """
        # Check if we have recently assessed this token
        current_time = time.time()
        if (not force_refresh and 
            token_address in self.last_risk_assessment and
            current_time - self.last_risk_assessment[token_address] < self.config["risk_assessment_refresh_interval"]):
            # Return cached assessment
            return self.risk_assessments.get(token_address, {})
        
        # We need to get token data from the knowledge base
        if self.knowledge_base is None:
            logger.error("Cannot assess token risk: no knowledge base available")
            return {}
        
        try:
            # Record start time for performance measurement
            start_time = time.time()
            
            # Report that assessment is starting
            await self.send_confidence_report(
                task_name=f"risk_assessment_start_{token_address}",
                confidence=5.0,
                details=f"Starting risk assessment for {token_address}"
            )
            
            # Check if token is blacklisted
            if token_address in self.config["blacklisted_tokens"]:
                assessment = {
                    "token_address": token_address,
                    "overall_risk": RiskLevel.CRITICAL,
                    "overall_risk_score": 1.0,
                    "is_blacklisted": True,
                    "risk_factors": [
                        {
                            "type": "blacklist",
                            "level": RiskLevel.CRITICAL,
                            "score": 1.0,
                            "details": "Token is blacklisted"
                        }
                    ],
                    "assessment_time": datetime.utcnow().isoformat()
                }
                
                # Store results
                self.risk_assessments[token_address] = assessment
                self.last_risk_assessment[token_address] = current_time
                
                return assessment
            
            # Get token data from knowledge base
            entry = await self.knowledge_base.get_entry(f"token:{token_address}")
            if not entry:
                logger.warning(f"No token data found for {token_address}")
                return {}
            
            token_data = entry.data
            
            # Check if developer is blacklisted
            developer_address = token_data.get("developer_address", "unknown")
            if developer_address in self.config["blacklisted_developers"]:
                assessment = {
                    "token_address": token_address,
                    "overall_risk": RiskLevel.CRITICAL,
                    "overall_risk_score": 1.0,
                    "is_blacklisted": True,
                    "risk_factors": [
                        {
                            "type": "blacklist",
                            "level": RiskLevel.CRITICAL,
                            "score": 1.0,
                            "details": "Developer is blacklisted"
                        }
                    ],
                    "assessment_time": datetime.utcnow().isoformat()
                }
                
                # Store results
                self.risk_assessments[token_address] = assessment
                self.last_risk_assessment[token_address] = current_time
                
                return assessment
            
            # Assess various risk factors
            risk_factors = []
            
            # 1. SPL token compliance check
            if self.spl_checker:
                spl_compliance = await self._assess_spl_compliance(token_address, token_data)
                if spl_compliance:
                    risk_factors.append(spl_compliance)
            
            # 2. Contract vulnerabilities risk
            contract_risk = await self._assess_contract_risk(token_data)
            if contract_risk:
                risk_factors.append(contract_risk)
            
            # 3. Program vulnerability risk
            program_risk = await self._assess_program_vulnerability(token_data)
            if program_risk:
                risk_factors.append(program_risk)
            
            # 4. Enhanced liquidity risk assessment using Jupiter
            liquidity_risk = await self._assess_liquidity_risk(token_address, token_data)
            if liquidity_risk:
                risk_factors.append(liquidity_risk)
            
            # 5. Liquidity fragility risk (new)
            fragility_risk = await self._assess_liquidity_fragility(token_address, token_data)
            if fragility_risk:
                risk_factors.append(fragility_risk)
            
            # 6. Owner concentration risk
            concentration_risk = await self._assess_concentration_risk(token_data)
            if concentration_risk:
                risk_factors.append(concentration_risk)
            
            # 7. Oracle price comparison using Switchboard
            oracle_risk = await self._assess_oracle_price_deviation(token_address, token_data)
            if oracle_risk:
                risk_factors.append(oracle_risk)
            
            # 8. Helius metadata verification (if available)
            if self.helius_client:
                metadata_risk = await self._assess_metadata_verification(token_address, token_data)
                if metadata_risk:
                    risk_factors.append(metadata_risk)
            
            # 9. ML-based pattern anomaly detection
            anomaly_risk = await self._assess_anomaly_patterns(token_data)
            if anomaly_risk:
                risk_factors.append(anomaly_risk)
            
            # 10. Developer reputation risk (enhanced with Helius if available)
            dev_risk = await self._assess_developer_risk(token_address, token_data)
            if dev_risk:
                risk_factors.append(dev_risk)
            
            # Calculate overall risk score with weighting
            total_weight = 0
            weighted_score = 0
            
            # Define risk factor weights
            factor_weights = {
                RiskFactorType.SPL_COMPLIANCE: 0.20,
                RiskFactorType.CONTRACT_RISK: 0.15,
                RiskFactorType.PROGRAM_VULNERABILITY: 0.15,
                RiskFactorType.LIQUIDITY_RISK: 0.10,
                RiskFactorType.LIQUIDITY_FRAGILITY: 0.10,
                RiskFactorType.OWNER_CONCENTRATION: 0.10,
                RiskFactorType.ORACLE_PRICE_DEVIATION: 0.05,
                RiskFactorType.HELIUS_METADATA: 0.05,
                "default": 0.05
            }
            
            for factor in risk_factors:
                factor_type = factor.get("type")
                score = factor.get("score", 0)
                
                # Get weight for this factor type or use default
                weight = factor_weights.get(factor_type, factor_weights["default"])
                
                weighted_score += score * weight
                total_weight += weight
            
            # Calculate overall risk score (normalized to 0-1)
            if total_weight > 0:
                overall_risk_score = weighted_score / total_weight
            else:
                overall_risk_score = 0.0
            
            # Determine overall risk level
            if overall_risk_score >= 0.75:
                overall_risk = RiskLevel.CRITICAL
            elif overall_risk_score >= 0.5:
                overall_risk = RiskLevel.HIGH
            elif overall_risk_score >= 0.25:
                overall_risk = RiskLevel.MEDIUM
            else:
                overall_risk = RiskLevel.LOW
            
            # Report confidence in assessment
            await self.send_confidence_report(
                task_name=f"risk_assessment_{token_address}",
                confidence=8.5,
                details=f"Completed risk assessment for {token_address} with {len(risk_factors)} risk factors"
            )
            
            # Calculate assessment duration
            assessment_duration = time.time() - start_time
            
            # Compile assessment result
            risk_assessment = {
                "token_address": token_address,
                "token_name": token_data.get("name", "Unknown"),
                "token_symbol": token_data.get("symbol", "Unknown"),
                "overall_risk": overall_risk,
                "overall_risk_score": overall_risk_score,
                "risk_factors": risk_factors,
                "is_blacklisted": False,
                "assessment_time": datetime.utcnow().isoformat(),
                "assessment_duration_seconds": assessment_duration,
                "enhancements_used": [
                    "spl_compliance_check" if self.spl_checker else None,
                    "jupiter_liquidity_analysis" if self.jupiter_client else None,
                    "helius_metadata" if self.helius_client else None,
                    "switchboard_oracle" if oracle_risk else None,
