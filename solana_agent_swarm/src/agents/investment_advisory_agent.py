"""
Investment Advisory Agent for the Solana Token Analysis Agent Swarm.
Aggregates data from pattern analysis and risk assessment to provide
investment recommendations and opportunities.
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


class ConfidenceLevel:
    """Constants for investment confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InvestmentAdvisoryAgent(Agent):
    """
    Agent responsible for providing investment recommendations based on
    pattern analysis and risk assessment.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Investment Advisory Agent.
        
        Args:
            agent_id: Optional agent ID. If not provided, one will be generated.
            knowledge_base: Optional knowledge base instance to use.
                If not provided, the agent will expect to receive one
                from the orchestrator after initialization.
            config: Optional configuration for advisory parameters.
        """
        agent_id = agent_id or f"investment-advisor-{str(uuid.uuid4())[:8]}"
        super().__init__(agent_id=agent_id, agent_type="InvestmentAdvisoryAgent")
        
        self.knowledge_base = knowledge_base
        
        # Default configuration - can be overridden by provided config
        self.config = {
            "high_confidence_min_pattern_score": 10,  # Minimum pattern score for high confidence
            "medium_confidence_min_pattern_score": 5,  # Minimum pattern score for medium confidence
            "max_risk_score_for_high_confidence": 0.3,  # Maximum risk score for high confidence
            "max_risk_score_for_medium_confidence": 0.5,  # Maximum risk score for medium confidence
            "pattern_weight": 0.6,  # Weight for pattern analysis in final scoring
            "risk_weight": 0.4,  # Weight for risk assessment in final scoring
            "opportunity_refresh_interval": 300,  # Seconds between opportunity refreshes
            "opportunity_expiry": 86400,  # Seconds until an opportunity is considered stale (1 day)
            "max_opportunities_to_track": 100,  # Maximum number of opportunities to track
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Investment opportunity tracking
        self.opportunities = {}  # token_address -> opportunity data
        self.last_opportunity_update = {}  # token_address -> timestamp of last update
        
        # Dependencies
        self.pattern_analysis_agent_id = None
        self.risk_assessment_agent_id = None
        
        logger.info(f"Investment Advisory Agent {self.id} initialized")
    
    async def _initialize(self) -> None:
        """Initialize the agent."""
        # No special initialization needed beyond the base class
        logger.info(f"Investment Advisory Agent {self.id} ready")
    
    async def _handle_message(self, message: Message) -> None:
        """
        Handle incoming messages.
        
        Args:
            message: The message to handle.
        """
        logger.debug(f"Investment Advisory Agent {self.id} received message: {message.type.value}")
        
        if message.type == MessageType.EVENT:
            await self._handle_event(message)
        elif message.type == MessageType.COMMAND:
            await self._handle_command(message)
        elif message.type == MessageType.QUERY:
            await self._handle_query(message)
        elif message.type == MessageType.RESPONSE:
            await self._handle_response(message)
        else:
            logger.warning(f"Unsupported message type: {message.type.value}")
    
    async def _handle_event(self, message: Message) -> None:
        """
        Handle event messages.
        
        Args:
            message: The event message.
        """
        event_type = message.content.get("event_type")
        
        if event_type == "critical_pattern_detected":
            # Handle critical pattern event from pattern analysis agent
            token_address = message.content.get("token_address")
            patterns = message.content.get("patterns", [])
            
            if token_address:
                logger.info(f"Critical pattern detected for {token_address}, evaluating investment opportunity")
                
                # Request risk assessment for this token
                if self.risk_assessment_agent_id:
                    query = Message(
                        msg_type=MessageType.QUERY,
                        sender_id=self.id,
                        target_id=self.risk_assessment_agent_id,
                        content={
                            "query_type": "get_token_risk",
                            "token_address": token_address
                        }
                    )
                    await self.send_message(query)
        
        elif event_type == "high_risk_token_detected":
            # Handle high risk event from risk assessment agent
            token_address = message.content.get("token_address")
            risk_level = message.content.get("risk_level")
            
            if token_address and token_address in self.opportunities:
                logger.info(f"High risk detected for {token_address}, removing from opportunities")
                
                # Remove from opportunities
                if token_address in self.opportunities:
                    del self.opportunities[token_address]
                if token_address in self.last_opportunity_update:
                    del self.last_opportunity_update[token_address]
                
                # Update knowledge base entry if exists
                if self.knowledge_base:
                    entry = await self.knowledge_base.get_entry(f"opportunity:{token_address}")
                    if entry:
                        # Mark as expired
                        await self.knowledge_base.add_entry(
                            entry_id=f"opportunity:{token_address}",
                            entry_type=EntryType.INVESTMENT_OPPORTUNITY,
                            data={
                                "token_address": token_address,
                                "status": "expired",
                                "reason": f"High risk detected: {risk_level}",
                                "updated_at": datetime.utcnow().isoformat()
                            },
                            metadata={
                                "active": False,
                                "high_risk": True
                            },
                            source_agent_id=self.id,
                            tags=["opportunity", "expired", "high_risk"]
                        )
    
    async def _handle_command(self, message: Message) -> None:
        """
        Handle command messages.
        
        Args:
            message: The command message.
        """
        command = message.content.get("command")
        
        if command == "evaluate_opportunity":
            # Handle command to evaluate investment opportunity for a token
            token_address = message.content.get("token_address")
            force_refresh = message.content.get("force_refresh", False)
            
            if token_address:
                try:
                    # Evaluate investment opportunity
                    opportunity = await self._evaluate_investment_opportunity(token_address, force_refresh)
                    
                    # Send response with opportunity
                    response = Message(
                        msg_type=MessageType.RESPONSE,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "token_address": token_address,
                            "opportunity": opportunity
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
                            "error": f"Failed to evaluate investment opportunity: {str(e)}",
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
        
        elif command == "set_dependencies":
            # Set dependencies on other agents
            pattern_analysis_agent_id = message.content.get("pattern_analysis_agent_id")
            risk_assessment_agent_id = message.content.get("risk_assessment_agent_id")
            
            if pattern_analysis_agent_id:
                self.pattern_analysis_agent_id = pattern_analysis_agent_id
                logger.info(f"Set pattern analysis agent dependency to {pattern_analysis_agent_id}")
            
            if risk_assessment_agent_id:
                self.risk_assessment_agent_id = risk_assessment_agent_id
                logger.info(f"Set risk assessment agent dependency to {risk_assessment_agent_id}")
            
            # Send acknowledgement
            response = Message(
                msg_type=MessageType.RESPONSE,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "status": "success",
                    "pattern_analysis_agent_id": self.pattern_analysis_agent_id,
                    "risk_assessment_agent_id": self.risk_assessment_agent_id
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
        
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
        
        if query_type == "get_investment_opportunity":
            # Handle query for investment opportunity for a specific token
            token_address = message.content.get("token_address")
            
            if token_address:
                # Get opportunity from cache or evaluate if needed
                if token_address in self.opportunities:
                    opportunity = self.opportunities[token_address]
                else:
                    opportunity = await self._evaluate_investment_opportunity(token_address)
                
                # Send response with opportunity
                response = Message(
                    msg_type=MessageType.RESPONSE,
                    sender_id=self.id,
                    target_id=message.sender_id,
                    content={
                        "token_address": token_address,
                        "opportunity": opportunity
                    },
                    correlation_id=message.correlation_id
                )
                await self.send_message(response)
        
        elif query_type == "get_top_opportunities":
            # Handle query for top investment opportunities
            count = message.content.get("count", 10)
            min_confidence = message.content.get("min_confidence", ConfidenceLevel.MEDIUM)
            
            # Get top opportunities
            top_opportunities = await self._get_top_opportunities(count, min_confidence)
            
            # Send response
            response = Message(
                msg_type=MessageType.RESPONSE,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "top_opportunities": top_opportunities
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
    
    async def _handle_response(self, message: Message) -> None:
        """
        Handle response messages, particularly from pattern analysis and risk assessment agents.
        
        Args:
            message: The response message.
        """
        # Check if this is a response from the pattern analysis agent
        if message.sender_id == self.pattern_analysis_agent_id:
            # Handle patterns response
            token_address = message.content.get("token_address")
            patterns = message.content.get("patterns", [])
            
            if token_address and patterns:
                logger.info(f"Received pattern analysis for {token_address}")
                
                # Store temporarily
                token_key = f"pattern:{token_address}"
                
                # Store in knowledge base with short TTL
                if self.knowledge_base:
                    await self.knowledge_base.add_entry(
                        entry_id=token_key,
                        entry_type=EntryType.PATTERN,
                        data={
                            "token_address": token_address,
                            "patterns": patterns,
                            "analysis_time": datetime.utcnow().isoformat()
                        },
                        ttl=300,  # 5 minutes TTL
                        source_agent_id=self.id,
                        tags=["temp", "pattern_response"]
                    )
                
                # If we have risk data, evaluate opportunity
                risk_key = f"risk:{token_address}"
                risk_entry = await self.knowledge_base.get_entry(risk_key) if self.knowledge_base else None
                
                if risk_entry:
                    # We have both pattern and risk data, evaluate opportunity
                    await self._evaluate_with_pattern_and_risk(
                        token_address,
                        patterns,
                        risk_entry.data
                    )
                else:
                    # Request risk assessment
                    if self.risk_assessment_agent_id:
                        query = Message(
                            msg_type=MessageType.QUERY,
                            sender_id=self.id,
                            target_id=self.risk_assessment_agent_id,
                            content={
                                "query_type": "get_token_risk",
                                "token_address": token_address
                            }
                        )
                        await self.send_message(query)
        
        # Check if this is a response from the risk assessment agent
        elif message.sender_id == self.risk_assessment_agent_id:
            # Handle risk assessment response
            token_address = message.content.get("token_address")
            risk_assessment = message.content.get("risk_assessment", {})
            
            if token_address and risk_assessment:
                logger.info(f"Received risk assessment for {token_address}")
                
                # Get pattern data if available
                pattern_key = f"pattern:{token_address}"
                pattern_entry = await self.knowledge_base.get_entry(pattern_key) if self.knowledge_base else None
                
                if pattern_entry:
                    # We have both pattern and risk data, evaluate opportunity
                    await self._evaluate_with_pattern_and_risk(
                        token_address,
                        pattern_entry.data.get("patterns", []),
                        risk_assessment
                    )
                else:
                    # Request pattern analysis
                    if self.pattern_analysis_agent_id:
                        query = Message(
                            msg_type=MessageType.QUERY,
                            sender_id=self.id,
                            target_id=self.pattern_analysis_agent_id,
                            content={
                                "query_type": "get_token_patterns",
                                "token_address": token_address
                            }
                        )
                        await self.send_message(query)
    
    async def _evaluate_with_pattern_and_risk(
        self,
        token_address: str,
        patterns: List[Dict[str, Any]],
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate investment opportunity using both pattern and risk data.
        
        Args:
            token_address: The token's address.
            patterns: List of detected patterns.
            risk_assessment: Risk assessment data.
            
        Returns:
            Investment opportunity data.
        """
        # Get token data from knowledge base
        token_data = {}
        if self.knowledge_base:
            token_entry = await self.knowledge_base.get_entry(f"token:{token_address}")
            if token_entry:
                token_data = token_entry.data
        
        # Calculate pattern score
        pattern_score = 0
        for pattern in patterns:
            severity = pattern.get("severity", "low")
            # Assign scores based on severity
            if severity == "critical":
                pattern_score += 3
            elif severity == "high":
                pattern_score += 2
            elif severity == "medium":
                pattern_score += 1
        
        # Get risk score
        risk_score = risk_assessment.get("overall_risk_score", 0.5)
        
        # Calculate weighted score
        weighted_score = (
            self.config["pattern_weight"] * min(pattern_score / 10, 1.0) +
            self.config["risk_weight"] * (1.0 - risk_score)
        )
        
        # Determine confidence level
        confidence = ConfidenceLevel.LOW
        if (pattern_score >= self.config["high_confidence_min_pattern_score"] and 
            risk_score <= self.config["max_risk_score_for_high_confidence"]):
            confidence = ConfidenceLevel.HIGH
        elif (pattern_score >= self.config["medium_confidence_min_pattern_score"] and 
              risk_score <= self.config["max_risk_score_for_medium_confidence"]):
            confidence = ConfidenceLevel.MEDIUM
        
        # Skip if high risk
        if risk_assessment.get("overall_risk") in ["high", "critical"]:
            confidence = ConfidenceLevel.LOW
        
        # Create opportunity data
        opportunity = {
            "token_address": token_address,
            "token_name": token_data.get("name", "Unknown"),
            "token_symbol": token_data.get("symbol", "Unknown"),
            "confidence": confidence,
            "weighted_score": weighted_score,
            "pattern_score": pattern_score,
            "risk_score": risk_score,
            "significant_patterns": [
                p for p in patterns 
                if p.get("severity") in ["high", "critical"]
            ],
            "risk_assessment_summary": {
                "overall_risk": risk_assessment.get("overall_risk"),
                "overall_risk_score": risk_assessment.get("overall_risk_score"),
                "is_blacklisted": risk_assessment.get("is_blacklisted", False)
            },
            "market_data": token_data.get("market", {}),
            "evaluation_time": datetime.utcnow().isoformat()
        }
        
        # Store in cache
        current_time = time.time()
        self.opportunities[token_address] = opportunity
        self.last_opportunity_update[token_address] = current_time
        
        # Store in knowledge base if medium or high confidence
        if confidence in [ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH] and self.knowledge_base:
            await self.knowledge_base.add_entry(
                entry_id=f"opportunity:{token_address}",
                entry_type=EntryType.INVESTMENT_OPPORTUNITY,
                data=opportunity,
                metadata={
                    "active": True,
                    "high_confidence": confidence == ConfidenceLevel.HIGH
                },
                source_agent_id=self.id,
                tags=["opportunity", "active", confidence]
            )
            
            # Notify about high confidence opportunities
            if confidence == ConfidenceLevel.HIGH:
                event_msg = Message(
                    msg_type=MessageType.EVENT,
                    sender_id=self.id,
                    content={
                        "event_type": "high_confidence_opportunity",
                        "token_address": token_address,
                        "token_name": token_data.get("name", "Unknown"),
                        "token_symbol": token_data.get("symbol", "Unknown"),
                        "weighted_score": weighted_score,
                        "discovery_time": datetime.utcnow().isoformat()
                    }
                )
                await self.send_message(event_msg)
        
        return opportunity
    
    async def _evaluate_investment_opportunity(
        self,
        token_address: str,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate investment opportunity for a specific token.
        
        Args:
            token_address: The token's address.
            force_refresh: Whether to force re-evaluation.
            
        Returns:
            Investment opportunity data.
        """
        # Check if we have recently evaluated this token
        current_time = time.time()
        if (not force_refresh and 
            token_address in self.last_opportunity_update and
            current_time - self.last_opportunity_update[token_address] < self.config["opportunity_refresh_interval"]):
            # Return cached opportunity
            return self.opportunities.get(token_address, {})
        
        # Request pattern analysis
        if self.pattern_analysis_agent_id:
            query = Message(
                msg_type=MessageType.QUERY,
                sender_id=self.id,
                target_id=self.pattern_analysis_agent_id,
                content={
                    "query_type": "get_token_patterns",
                    "token_address": token_address
                }
            )
            await self.send_message(query)
        
        # Request risk assessment
        if self.risk_assessment_agent_id:
            query = Message(
                msg_type=MessageType.QUERY,
                sender_id=self.id,
                target_id=self.risk_assessment_agent_id,
                content={
                    "query_type": "get_token_risk",
                    "token_address": token_address
                }
            )
            await self.send_message(query)
        
        # Return existing opportunity if available
        # (the full evaluation will happen asynchronously when responses arrive)
        return self.opportunities.get(token_address, {})
    
    async def _get_top_opportunities(
        self,
        count: int = 10,
        min_confidence: str = ConfidenceLevel.MEDIUM
    ) -> List[Dict[str, Any]]:
        """
        Get top investment opportunities.
        
        Args:
            count: Maximum number of opportunities to return.
            min_confidence: Minimum confidence level to include.
            
        Returns:
            List of top investment opportunities.
        """
        # In a real implementation, we would query the knowledge base
        # For this example, we'll use our cached opportunities
        
        # Filter by minimum confidence level
        confidence_levels = {
            ConfidenceLevel.LOW: 1,
            ConfidenceLevel.MEDIUM: 2,
            ConfidenceLevel.HIGH: 3
        }
        min_confidence_value = confidence_levels.get(min_confidence, 1)
        
        # Filter and sort opportunities
        current_time = time.time()
        filtered_opportunities = []
        
        for token_address, opportunity in self.opportunities.items():
            # Check if opportunity is stale
            if token_address in self.last_opportunity_update:
                age = current_time - self.last_opportunity_update[token_address]
                if age > self.config["opportunity_expiry"]:
                    continue  # Skip stale opportunities
            
            # Check confidence level
            confidence = opportunity.get("confidence", ConfidenceLevel.LOW)
            if confidence_levels.get(confidence, 0) >= min_confidence_value:
                filtered_opportunities.append(opportunity)
        
        # Sort by weighted score in descending order
        filtered_opportunities.sort(
            key=lambda x: x.get("weighted_score", 0),
            reverse=True
        )
        
        # Return top opportunities
        return filtered_opportunities[:count]
    
    async def _cleanup(self) -> None:
        """Clean up resources when stopping the agent."""
        # Clean up stale opportunities
        current_time = time.time()
        stale_opportunities = []
        
        for token_address, timestamp in self.last_opportunity_update.items():
            age = current_time - timestamp
            if age > self.config["opportunity_expiry"]:
                stale_opportunities.append(token_address)
        
        for token_address in stale_opportunities:
            if token_address in self.opportunities:
                del self.opportunities[token_address]
            if token_address in self.last_opportunity_update:
                del self.last_opportunity_update[token_address]
        
        logger.info(f"Investment Advisory Agent {self.id} cleaned up, removed {len(stale_opportunities)} stale opportunities")
