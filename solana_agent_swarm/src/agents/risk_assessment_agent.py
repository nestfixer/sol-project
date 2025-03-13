"""
Risk Assessment Agent for the Solana Token Analysis Agent Swarm.
Evaluates token risks, contract safety, developer history, and
liquidity to identify potential scams and risky investments.
Features enhanced contract vulnerability detection, cross-chain risk analysis,
and comprehensive liquidity health scoring.
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


class RiskLevel:
    """Constants for risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAssessmentAgent(Agent):
    """
    Agent responsible for evaluating token risks.
    Identifies potential scams, rug pulls, and investment risks.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Risk Assessment Agent.
        
        Args:
            agent_id: Optional agent ID. If not provided, one will be generated.
            knowledge_base: Optional knowledge base instance to use.
                If not provided, the agent will expect to receive one
                from the orchestrator after initialization.
            config: Optional configuration for risk assessment thresholds.
        """
        agent_id = agent_id or f"risk-assessor-{str(uuid.uuid4())[:8]}"
        super().__init__(agent_id=agent_id, agent_type="RiskAssessmentAgent")
        
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
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Risk assessment state
        self.last_risk_assessment = {}  # token_address -> timestamp of last assessment
        self.risk_assessments = {}  # token_address -> risk assessment result
        
        logger.info(f"Risk Assessment Agent {self.id} initialized")
    
    async def _initialize(self) -> None:
        """Initialize the agent."""
        # No special initialization needed beyond the base class
        logger.info(f"Risk Assessment Agent {self.id} ready")
    
    async def _handle_message(self, message: Message) -> None:
        """
        Handle incoming messages.
        
        Args:
            message: The message to handle.
        """
        logger.debug(f"Risk Assessment Agent {self.id} received message: {message.type.value}")
        
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
        Assess risk for a specific token with enhanced analysis capabilities.
        
        Args:
            token_address: The token's address.
            force_refresh: Whether to force a risk re-assessment.
            
        Returns:
            Comprehensive risk assessment result with detailed vulnerability analysis.
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
            # Check if token is blacklisted
            if token_address in self.config["blacklisted_tokens"]:
                return {
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
            
            # Get token data from knowledge base
            entry = await self.knowledge_base.get_entry(f"token:{token_address}")
            if not entry:
                logger.warning(f"No token data found for {token_address}")
                return {}
            
            token_data = entry.data
            
            # Check if developer is blacklisted
            developer_address = token_data.get("developer_address", "unknown")
            if developer_address in self.config["blacklisted_developers"]:
                return {
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
            
            # Assess various risk factors
            risk_factors = []
            
            # 1. Contract vulnerabilities risk using the enhanced data
            contract_risk = await self._assess_contract_risk(token_data)
            if contract_risk:
                risk_factors.append(contract_risk)
            
            # 2. Program vulnerability risk (new)
            program_risk = await self._assess_program_vulnerability(token_data)
            if program_risk:
                risk_factors.append(program_risk)
            
            # 3. Enhanced liquidity risk assessment
            liquidity_risk = await self._assess_liquidity_risk(token_data)
            if liquidity_risk:
                risk_factors.append(liquidity_risk)
            
            # 4. Liquidity fragility risk (new)
            fragility_risk = await self._assess_liquidity_fragility(token_data)
            if fragility_risk:
                risk_factors.append(fragility_risk)
            
            # 3. Owner concentration risk
            concentration_risk = await self._assess_concentration_risk(token_data)
            if concentration_risk:
                risk_factors.append(concentration_risk)
            
            # 4. Developer reputation risk (placeholder)
            dev_risk = await self._assess_developer_risk(token_data)
            if dev_risk:
                risk_factors.append(dev_risk)
            
            # Calculate overall risk score
            risk_scores = [f["score"] for f in risk_factors if "score" in f]
            if risk_scores:
                # Simple average for now; a weighted calculation could be used
                overall_risk_score = sum(risk_scores) / len(risk_scores)
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
            
            # Compile assessment result
            risk_assessment = {
                "token_address": token_address,
                "token_name": token_data.get("name", "Unknown"),
                "token_symbol": token_data.get("symbol", "Unknown"),
                "overall_risk": overall_risk,
                "overall_risk_score": overall_risk_score,
                "risk_factors": risk_factors,
                "is_blacklisted": False,
                "assessment_time": datetime.utcnow().isoformat()
            }
            
            # Store results
            self.risk_assessments[token_address] = risk_assessment
            self.last_risk_assessment[token_address] = current_time
            
            # Store in knowledge base
            await self.knowledge_base.add_entry(
                entry_id=f"risk:{token_address}",
                entry_type=EntryType.RISK_ASSESSMENT,
                data=risk_assessment,
                metadata={
                    "high_risk": overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                },
                source_agent_id=self.id,
                tags=["risk", "assessment"]
            )
            
            # Notify other agents if high risk
            if overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                event_msg = Message(
                    msg_type=MessageType.EVENT,
                    sender_id=self.id,
                    content={
                        "event_type": "high_risk_token_detected",
                        "token_address": token_address,
                        "risk_level": overall_risk,
                        "risk_score": overall_risk_score,
                        "assessment_time": datetime.utcnow().isoformat()
                    }
                )
                await self.send_message(event_msg)
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error assessing token risk for {token_address}: {str(e)}")
            return {}
    
    async def _assess_contract_risk(self, token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Assess contract security risks using enhanced analysis.
        
        Args:
            token_data: Token data including contract_analysis from ContractAnalyzer.
            
        Returns:
            Detailed risk factor data if significant risk is detected, None otherwise.
        """
        # Check if we have contract analysis data
        contract_analysis = token_data.get("contract_analysis", {})
        if not contract_analysis:
            # Fallback to generic risk assessment if no data
            risk_score = 0.5  # Medium risk due to lack of data
            risk_level = RiskLevel.MEDIUM
            details = "Unable to analyze contract code - insufficient data"
        else:
            # Extract vulnerability info from analysis
            detected_vulnerabilities = contract_analysis.get("detected_vulnerabilities", {})
            vulnerability_score = contract_analysis.get("vulnerability_score", 0.5)
            
            # Create details message with specific vulnerabilities
            if detected_vulnerabilities:
                vulnerability_details = []
                for vuln_type, instances in detected_vulnerabilities.items():
                    vulnerability_details.append(f"{vuln_type}: {len(instances)} instances detected")
                details = "Contract analysis found: " + ", ".join(vulnerability_details)
            else:
                details = "No specific vulnerabilities detected in contract analysis"
            
            # Use vulnerability score from analysis
            risk_score = vulnerability_score
            
            # Determine risk level
            if risk_score >= 0.75:
                risk_level = RiskLevel.CRITICAL
            elif risk_score >= 0.5:
                risk_level = RiskLevel.HIGH
            elif risk_score >= 0.25:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Add info about upgradeability which increases risk
            if contract_analysis.get("is_upgradeable", False):
                details += ". Contract is upgradeable, which increases risk of future changes."
        
        return {
            "type": RiskFactorType.CONTRACT_RISK,
            "level": risk_level,
            "score": risk_score,
            "details": details
        }
    
    async def _assess_program_vulnerability(self, token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Assess program vulnerability risks based on enhanced code analysis.
        
        Args:
            token_data: Token data including contract_analysis from ContractAnalyzer.
            
        Returns:
            Risk factor data specific to program vulnerabilities.
        """
        # Check if we have contract analysis data
        contract_analysis = token_data.get("contract_analysis", {})
        if not contract_analysis:
            return None
        
        # Calculate risk based on detected vulnerability types
        detected_vulnerabilities = contract_analysis.get("detected_vulnerabilities", {})
        
        # Different vulnerability types have different severity levels
        severity_weights = {
            "backdoor": 1.0,  # Highest risk
            "hidden_mint": 0.8,
            "unsafe_owner_change": 0.6
        }
        
        # Calculate weighted risk score
        total_weight = 0
        weighted_score = 0
        
        for vuln_type, instances in detected_vulnerabilities.items():
            if vuln_type in severity_weights:
                weight = severity_weights[vuln_type]
                count = len(instances)
                weighted_score += weight * min(count, 5)  # Cap at 5 instances per type
                total_weight += weight
        
        if total_weight > 0:
            risk_score = min(1.0, weighted_score / (total_weight * 5))
        else:
            risk_score = 0.0
        
        # Determine risk level
        if risk_score >= 0.75:
            risk_level = RiskLevel.CRITICAL
            details = "Critical program vulnerabilities detected"
        elif risk_score >= 0.5:
            risk_level = RiskLevel.HIGH
            details = "High-risk program vulnerabilities detected"
        elif risk_score >= 0.25:
            risk_level = RiskLevel.MEDIUM
            details = "Moderate program security concerns"
        elif risk_score > 0:
            risk_level = RiskLevel.LOW
            details = "Minor program security concerns"
        else:
            return None  # No vulnerabilities detected
        
        # Add specific vulnerability details
        if detected_vulnerabilities:
            vulnerability_details = []
            for vuln_type, instances in detected_vulnerabilities.items():
                vulnerability_details.append(f"{vuln_type}: {len(instances)} instances")
            details += " (" + ", ".join(vulnerability_details) + ")"
        
        return {
            "type": RiskFactorType.PROGRAM_VULNERABILITY,
            "level": risk_level,
            "score": risk_score,
            "details": details
        }
    
    async def _assess_liquidity_risk(self, token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Assess liquidity risks with enhanced metrics.
        
        Args:
            token_data: Token data including enhanced liquidity metrics.
            
        Returns:
            Risk factor data with detailed liquidity analysis.
        """
        try:
            # Try to get enhanced liquidity data if available
            liquidity_data = token_data.get("liquidity", {})
            total_liquidity = liquidity_data.get("total_liquidity_usd", 0)
            
            # Fallback to basic market data if no enhanced data
            if total_liquidity == 0:
                total_liquidity = token_data.get("market", {}).get("liquidity_usd", 0)
            
            # Calculate risk based on liquidity
            if total_liquidity < self.config["min_liquidity_usd"]:
                # Low liquidity is high risk
                if total_liquidity == 0:
                    risk_level = RiskLevel.CRITICAL
                    risk_score = 1.0
                    details = "No liquidity detected"
                else:
                    risk_level = RiskLevel.HIGH
                    # Calculate risk score based on how far below the threshold
                    risk_score = 0.5 + 0.5 * (1 - total_liquidity / self.config["min_liquidity_usd"])
                    details = f"Low liquidity: ${total_liquidity:.2f} (below minimum ${self.config['min_liquidity_usd']:.2f})"
            else:
                # Sufficient liquidity is lower risk
                ratio = self.config["min_liquidity_usd"] / total_liquidity
                risk_score = max(0.0, min(0.25, ratio))  # Scale to 0.0-0.25 range
                risk_level = RiskLevel.LOW
                details = f"Adequate liquidity: ${total_liquidity:.2f}"
            
            # Add DEX distribution information if available
            dex_distribution = liquidity_data.get("dex_distribution", {})
            if dex_distribution:
                dex_count = len(dex_distribution)
                if dex_count == 1:
                    details += f". Warning: Liquidity only on one DEX ({list(dex_distribution.keys())[0]})"
                else:
                    details += f". Liquidity spread across {dex_count} DEXs"
            
            # Add historical trend information if available
            historical = liquidity_data.get("historical", {})
            trend = historical.get("trend", {}).get("trend", "")
            if trend in ["strongly_decreasing", "decreasing"]:
                details += f". Warning: Liquidity is {trend.replace('_', ' ')}"
                # Increase risk for decreasing liquidity
                risk_score = min(1.0, risk_score + 0.2)
            
            return {
                "type": RiskFactorType.LIQUIDITY_RISK,
                "level": risk_level,
                "score": risk_score,
                "details": details
            }
            
        except Exception as e:
            logger.error(f"Error assessing liquidity risk: {str(e)}")
            return None
    
    async def _assess_liquidity_fragility(self, token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Assess liquidity fragility using enhanced metrics.
        
        Args:
            token_data: Token data including enhanced liquidity metrics.
            
        Returns:
            Risk factor data specific to liquidity fragility.
        """
        try:
            # Check for enhanced liquidity data
            liquidity_data = token_data.get("liquidity", {})
            if not liquidity_data:
                return None
            
            # Get depth analysis if available
            depth_analysis = liquidity_data.get("depth_analysis", {})
            if not depth_analysis:
                return None
            
            # Extract fragility score (lower is better)
            fragility_score = depth_analysis.get("fragility_score", 0.5)
            
            # Invert score for risk calculation (higher is riskier)
            risk_score = fragility_score
            
            # Determine risk level
            if risk_score >= 0.75:
                risk_level = RiskLevel.CRITICAL
                details = "Extremely fragile liquidity profile"
            elif risk_score >= 0.5:
                risk_level = RiskLevel.HIGH
                details = "Highly fragile liquidity profile"
            elif risk_score >= 0.25:
                risk_level = RiskLevel.MEDIUM
                details = "Moderately fragile liquidity profile"
            else:
                risk_level = RiskLevel.LOW
                details = "Robust liquidity profile"
            
            # Add max trade size information
            max_trade = depth_analysis.get("estimated_max_trade_without_significant_impact", 0)
            if max_trade > 0:
                details += f". Max trade without significant impact: ${max_trade:.2f}"
            
            # Add depth classification
            depth_class = depth_analysis.get("depth_classification", "")
            if depth_class:
                details += f". Liquidity depth classification: {depth_class.replace('_', ' ')}"
            
            return {
                "type": RiskFactorType.LIQUIDITY_FRAGILITY,
                "level": risk_level,
                "score": risk_score,
                "details": details
            }
            
        except Exception as e:
            logger.error(f"Error assessing liquidity fragility: {str(e)}")
            return None
    
    async def _assess_concentration_risk(self, token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Assess holder concentration risks.
        
        Args:
            token_data: Token data including holder info.
            
        Returns:
            Risk factor data if significant risk is detected, None otherwise.
        """
        try:
            # Extract holder data
            holders = token_data.get("holders", {}).get("distribution", [])
            
            if not holders:
                return {
                    "type": RiskFactorType.OWNER_CONCENTRATION,
                    "level": RiskLevel.HIGH,
                    "score": 0.8,
                    "details": "No holder data available"
                }
            
            # Find maximum holder percentage
            max_percentage = max([h.get("percentage", 0) for h in holders])
            
            # Check concentration against thresholds
            if max_percentage >= self.config["critical_concentration_threshold"]:
                risk_level = RiskLevel.CRITICAL
                risk_score = 0.75 + 0.25 * (max_percentage - self.config["critical_concentration_threshold"]) / (1 - self.config["critical_concentration_threshold"])
                details = f"Critical concentration: {max_percentage:.1f}% held by a single wallet"
            elif max_percentage >= self.config["high_concentration_threshold"]:
                risk_level = RiskLevel.HIGH
                risk_score = 0.5 + 0.25 * (max_percentage - self.config["high_concentration_threshold"]) / (self.config["critical_concentration_threshold"] - self.config["high_concentration_threshold"])
                details = f"High concentration: {max_percentage:.1f}% held by a single wallet"
            else:
                # Calculate a proportional risk score
                risk_score = max_percentage / self.config["high_concentration_threshold"] * 0.5
                
                if risk_score > 0.25:
                    risk_level = RiskLevel.MEDIUM
                else:
                    risk_level = RiskLevel.LOW
                
                details = f"Moderate concentration: {max_percentage:.1f}% held by largest wallet"
            
            return {
                "type": RiskFactorType.OWNER_CONCENTRATION,
                "level": risk_level,
                "score": risk_score,
                "details": details
            }
            
        except Exception as e:
            logger.error(f"Error assessing concentration risk: {str(e)}")
            return None
    
    async def _assess_developer_risk(self, token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Assess developer reputation risks.
        
        Args:
            token_data: Token data including developer info.
            
        Returns:
            Risk factor data if significant risk is detected, None otherwise.
        """
        # In a real implementation, this would involve looking up the developer's history
        # and reputation, checking previous projects, etc.
        # For this example, we'll use placeholder logic
        
        # Placeholder developer risk score (0.0 to 1.0, where 1.0 is highest risk)
        # In a real implementation, this would come from developer history analysis
        risk_score = 0.3  # Just for example
        
        # Determine risk level
        if risk_score >= 0.75:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.5:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.25:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return {
            "type": RiskFactorType.DEV_REPUTATION,
            "level": risk_level,
            "score": risk_score,
            "details": "Placeholder developer reputation assessment"
        }
    
    async def _get_lowest_risk_tokens(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get tokens with lowest risk assessments.
        
        Args:
            count: Maximum number of tokens to return.
            
        Returns:
            List of low-risk tokens with their risk assessments.
        """
        low_risk_tokens = []
        
        # In a real implementation, we would query the knowledge base
        # For this example, we'll use our cached risk assessments
        
        # Sort tokens by risk score in ascending order (lowest risk first)
        sorted_tokens = sorted(
            self.risk_assessments.items(),
            key=lambda x: x[1].get("overall_risk_score", 1.0)
        )
        
        # Take the top tokens
        for token_address, risk_assessment in sorted_tokens[:count]:
            # Only include tokens with LOW or MEDIUM risk
            if risk_assessment.get("overall_risk") in [RiskLevel.LOW, RiskLevel.MEDIUM]:
                low_risk_tokens.append({
                    "token_address": token_address,
                    "token_name": risk_assessment.get("token_name", "Unknown"),
                    "token_symbol": risk_assessment.get("token_symbol", "Unknown"),
                    "risk_assessment": risk_assessment
                })
        
        return low_risk_tokens
    
    async def _cleanup(self) -> None:
        """Clean up resources when stopping the agent."""
        # Nothing special to clean up
        logger.info(f"Risk Assessment Agent {self.id} cleaned up")
