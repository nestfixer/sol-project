"""
Base Agent class that defines the common structure and capabilities
for all agents in the Solana Token Analysis Swarm.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from loguru import logger


class AgentStatus(Enum):
    """Enum representing the possible statuses of an agent."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class MessageType(Enum):
    """Enum representing the types of messages that can be sent between agents."""
    COMMAND = "command"
    DATA = "data"
    QUERY = "query"
    RESPONSE = "response"
    EVENT = "event"
    ERROR = "error"
    STATUS = "status"


class Message:
    """Standard message format for inter-agent communication."""
    
    def __init__(
        self, 
        msg_type: MessageType,
        sender_id: str,
        content: Dict[str, Any],
        target_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        priority: int = 1,
    ):
        self.id = str(uuid.uuid4())
        self.type = msg_type
        self.sender_id = sender_id
        self.target_id = target_id
        self.content = content
        self.timestamp = datetime.utcnow().isoformat()
        self.correlation_id = correlation_id or self.id
        self.priority = priority  # 1 (lowest) to 10 (highest)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender_id": self.sender_id,
            "target_id": self.target_id,
            "content": self.content,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "priority": self.priority,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a Message instance from a dictionary."""
        msg = cls(
            msg_type=MessageType(data["type"]),
            sender_id=data["sender_id"],
            content=data["content"],
            target_id=data.get("target_id"),
            correlation_id=data.get("correlation_id"),
            priority=data.get("priority", 1),
        )
        msg.id = data["id"]
        msg.timestamp = data["timestamp"]
        return msg


class Agent(ABC):
    """
    Base Agent class that all specialized agents inherit from.
    Provides common functionality for messaging, status tracking,
    and lifecycle management.
    """
    
    def __init__(self, agent_id: str, agent_type: str):
        self.id = agent_id
        self.type = agent_type
        self.status = AgentStatus.INITIALIZING
        self.created_at = datetime.utcnow().isoformat()
        self.last_active = self.created_at
        
        # Message queues
        self.inbound_queue = asyncio.Queue()
        self.outbound_queue = asyncio.Queue()
        
        # Tracking sets
        self.subscriptions: Set[str] = set()
        self.dependencies: Set[str] = set()
        
        # Runtime tracking
        self.task_count = 0
        self.error_count = 0
        self.processed_message_count = 0
        
        logger.info(f"Agent {self.id} of type {self.type} initialized")
    
    async def start(self) -> None:
        """Start the agent's operation."""
        if self.status != AgentStatus.INITIALIZING and self.status != AgentStatus.STOPPED:
            logger.warning(f"Agent {self.id} is already running or in an incompatible state: {self.status}")
            return
        
        try:
            await self._initialize()
            self.status = AgentStatus.READY
            logger.info(f"Agent {self.id} initialized and ready")
            
            # Start the message processing task
            asyncio.create_task(self._process_messages())
            self.status = AgentStatus.RUNNING
            logger.info(f"Agent {self.id} is now running")
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.error_count += 1
            logger.error(f"Failed to start agent {self.id}: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Stop the agent's operation."""
        logger.info(f"Stopping agent {self.id}")
        self.status = AgentStatus.STOPPED
        # Additional cleanup could be implemented here
        await self._cleanup()
    
    async def send_message(self, message: Message) -> None:
        """Send a message to another agent through the outbound queue."""
        await self.outbound_queue.put(message)
        logger.debug(f"Agent {self.id} queued message {message.id} for sending")
    
    async def receive_message(self, message: Message) -> None:
        """Receive a message from another agent into the inbound queue."""
        await self.inbound_queue.put(message)
        logger.debug(f"Agent {self.id} received message {message.id} from {message.sender_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent."""
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status.value,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "task_count": self.task_count,
            "error_count": self.error_count,
            "processed_message_count": self.processed_message_count,
            "inbound_queue_size": self.inbound_queue.qsize(),
            "outbound_queue_size": self.outbound_queue.qsize(),
        }
    
    async def _process_messages(self) -> None:
        """Process messages from the inbound queue."""
        while self.status == AgentStatus.RUNNING:
            try:
                # Wait for the next message
                message = await self.inbound_queue.get()
                
                # Update activity timestamp
                self.last_active = datetime.utcnow().isoformat()
                
                # Process the message
                await self._handle_message(message)
                
                # Mark the task as done
                self.inbound_queue.task_done()
                self.processed_message_count += 1
                
            except asyncio.CancelledError:
                logger.info(f"Message processing for agent {self.id} was cancelled")
                break
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error processing message in agent {self.id}: {str(e)}")
    
    @abstractmethod
    async def _initialize(self) -> None:
        """
        Initialize the agent with any necessary setup.
        To be implemented by each specific agent.
        """
        pass
    
    @abstractmethod
    async def _handle_message(self, message: Message) -> None:
        """
        Handle an incoming message.
        To be implemented by each specific agent.
        """
        pass
    
    @abstractmethod
    async def _cleanup(self) -> None:
        """
        Clean up resources when the agent is stopping.
        To be implemented by each specific agent.
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.type} Agent: {self.id} ({self.status.value})"
