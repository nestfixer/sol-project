"""
Orchestrator Agent for the Solana Token Analysis Swarm.
Responsible for agent management, task distribution, and workflow coordination.
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Set, Any, Tuple

from loguru import logger

from .agent_base import Agent, AgentStatus, Message, MessageType


class OrchestratorAgent(Agent):
    """
    The Orchestrator Agent is the central coordinator of the agent swarm.
    It manages agent lifecycles, distributes tasks, and handles
    system-wide coordination.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Orchestrator Agent.
        
        Args:
            config_path: Optional path to a configuration file.
        """
        super().__init__(
            agent_id=f"orchestrator-{str(uuid.uuid4())[:8]}",
            agent_type="Orchestrator"
        )
        
        # Registry of all agents in the system
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        
        # Agent type registry for looking up agents by their type
        self.agent_type_registry: Dict[str, Set[str]] = {}
        
        # Task registry for tracking currently executing tasks
        self.task_registry: Dict[str, Dict[str, Any]] = {}
        
        # Workflow registry for tracking multi-step workflows
        self.workflow_registry: Dict[str, Dict[str, Any]] = {}
        
        # Configuration path
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        
        # Message routing table
        self.routing_table: Dict[str, List[str]] = {}
        
        logger.info(f"Orchestrator Agent {self.id} initialized")
    
    async def _initialize(self) -> None:
        """Initialize the Orchestrator Agent."""
        # Load configuration if a path was provided
        if self.config_path:
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load configuration from {self.config_path}: {str(e)}")
                self.config = {}
        
        # Initialize message routing table
        self.routing_table = self.config.get("routing", {})
        
        # Start the outbound message processor
        asyncio.create_task(self._process_outbound_messages())
        
        logger.info("Orchestrator Agent initialized")
    
    async def _handle_message(self, message: Message) -> None:
        """
        Handle incoming messages to the Orchestrator.
        
        Args:
            message: The incoming message to process.
        """
        logger.debug(f"Orchestrator handling message: {message.type.value} from {message.sender_id}")
        
        # Handle different message types
        if message.type == MessageType.STATUS:
            await self._handle_status_message(message)
        elif message.type == MessageType.COMMAND:
            await self._handle_command_message(message)
        elif message.type == MessageType.QUERY:
            await self._handle_query_message(message)
        elif message.type == MessageType.ERROR:
            await self._handle_error_message(message)
        else:
            # Route the message to its target if specified
            if message.target_id:
                await self._route_message(message)
            else:
                logger.warning(f"Received message with no target: {message.id}")
    
    async def _handle_status_message(self, message: Message) -> None:
        """
        Handle status update messages from agents.
        
        Args:
            message: The status message to process.
        """
        agent_id = message.sender_id
        status_data = message.content
        
        # Update agent registry with the latest status
        if agent_id in self.agent_registry:
            self.agent_registry[agent_id].update(status_data)
            logger.debug(f"Updated status for agent {agent_id}")
        else:
            logger.warning(f"Received status for unknown agent: {agent_id}")
    
    async def _handle_command_message(self, message: Message) -> None:
        """
        Handle command messages for the orchestrator.
        
        Args:
            message: The command message to process.
        """
        command = message.content.get("command")
        
        if command == "register_agent":
            await self._register_agent(message)
        elif command == "unregister_agent":
            await self._unregister_agent(message)
        elif command == "start_agent":
            await self._start_agent(message)
        elif command == "stop_agent":
            await self._stop_agent(message)
        elif command == "create_workflow":
            await self._create_workflow(message)
        else:
            logger.warning(f"Unknown command: {command}")
            # Send an error response
            response = Message(
                msg_type=MessageType.ERROR,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "error": f"Unknown command: {command}",
                    "original_message_id": message.id
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
    
    async def _handle_query_message(self, message: Message) -> None:
        """
        Handle query messages for information from the orchestrator.
        
        Args:
            message: The query message to process.
        """
        query_type = message.content.get("query_type")
        
        if query_type == "agent_status":
            await self._handle_agent_status_query(message)
        elif query_type == "agents_by_type":
            await self._handle_agents_by_type_query(message)
        elif query_type == "system_status":
            await self._handle_system_status_query(message)
        elif query_type == "task_status":
            await self._handle_task_status_query(message)
        else:
            logger.warning(f"Unknown query type: {query_type}")
            # Send an error response
            response = Message(
                msg_type=MessageType.ERROR,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "error": f"Unknown query type: {query_type}",
                    "original_message_id": message.id
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
    
    async def _handle_error_message(self, message: Message) -> None:
        """
        Handle error messages from agents.
        
        Args:
            message: The error message to process.
        """
        error_data = message.content
        agent_id = message.sender_id
        
        logger.error(f"Received error from agent {agent_id}: {error_data}")
        
        # Update agent status in registry
        if agent_id in self.agent_registry:
            self.agent_registry[agent_id]["error_count"] += 1
            self.agent_registry[agent_id]["last_error"] = error_data
            
            # Check if the agent is in a critical state
            if self.agent_registry[agent_id]["error_count"] > 10:
                logger.critical(f"Agent {agent_id} has reported too many errors. Considering restart.")
                # TODO: Implement agent restart logic
    
    async def _route_message(self, message: Message) -> None:
        """
        Route a message to its target agent.
        
        Args:
            message: The message to route.
        """
        target_id = message.target_id
        
        if target_id in self.agent_registry:
            # TODO: Implement actual message delivery to agents
            # This would involve some form of connection to the target agent
            logger.debug(f"Routing message {message.id} to agent {target_id}")
            
            # For now, we just queue it to be handled by the target agent
            # In a real implementation, this would involve some form of IPC/RPC
            # to the actual agent process
            pass
        else:
            logger.warning(f"Cannot route message to unknown agent: {target_id}")
            # Send error back to sender
            error_message = Message(
                msg_type=MessageType.ERROR,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "error": f"Unknown target agent: {target_id}",
                    "original_message_id": message.id
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(error_message)
    
    async def _register_agent(self, message: Message) -> None:
        """
        Register a new agent with the orchestrator.
        
        Args:
            message: The registration command message.
        """
        agent_data = message.content.get("agent_data", {})
        agent_id = agent_data.get("id")
        agent_type = agent_data.get("type")
        
        if not agent_id or not agent_type:
            logger.error("Invalid agent registration data")
            return
        
        # Register the agent
        self.agent_registry[agent_id] = agent_data
        
        # Add to type registry
        if agent_type not in self.agent_type_registry:
            self.agent_type_registry[agent_type] = set()
        self.agent_type_registry[agent_type].add(agent_id)
        
        logger.info(f"Registered agent {agent_id} of type {agent_type}")
        
        # Send confirmation
        response = Message(
            msg_type=MessageType.RESPONSE,
            sender_id=self.id,
            target_id=message.sender_id,
            content={
                "status": "success",
                "message": f"Agent {agent_id} registered successfully"
            },
            correlation_id=message.correlation_id
        )
        await self.send_message(response)
    
    async def _unregister_agent(self, message: Message) -> None:
        """
        Unregister an agent from the orchestrator.
        
        Args:
            message: The unregistration command message.
        """
        agent_id = message.content.get("agent_id")
        
        if agent_id in self.agent_registry:
            agent_type = self.agent_registry[agent_id].get("type")
            
            # Remove from registry
            del self.agent_registry[agent_id]
            
            # Remove from type registry
            if agent_type and agent_type in self.agent_type_registry:
                self.agent_type_registry[agent_type].discard(agent_id)
                if not self.agent_type_registry[agent_type]:
                    del self.agent_type_registry[agent_type]
            
            logger.info(f"Unregistered agent {agent_id}")
            
            # Send confirmation
            response = Message(
                msg_type=MessageType.RESPONSE,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "status": "success",
                    "message": f"Agent {agent_id} unregistered successfully"
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
        else:
            logger.warning(f"Attempted to unregister unknown agent: {agent_id}")
            
            # Send error
            response = Message(
                msg_type=MessageType.ERROR,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "error": f"Unknown agent: {agent_id}",
                    "original_message_id": message.id
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
    
    async def _start_agent(self, message: Message) -> None:
        """
        Start an agent that is registered with the orchestrator.
        
        Args:
            message: The start command message.
        """
        agent_id = message.content.get("agent_id")
        
        if agent_id in self.agent_registry:
            logger.info(f"Starting agent {agent_id}")
            
            # TODO: Implement actual agent starting logic
            # This would involve starting the agent process or thread
            
            # Update status in registry
            self.agent_registry[agent_id]["status"] = AgentStatus.RUNNING.value
            
            # Send confirmation
            response = Message(
                msg_type=MessageType.RESPONSE,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "status": "success",
                    "message": f"Agent {agent_id} started successfully"
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
        else:
            logger.warning(f"Attempted to start unknown agent: {agent_id}")
            
            # Send error
            response = Message(
                msg_type=MessageType.ERROR,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "error": f"Unknown agent: {agent_id}",
                    "original_message_id": message.id
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
    
    async def _stop_agent(self, message: Message) -> None:
        """
        Stop an agent that is registered with the orchestrator.
        
        Args:
            message: The stop command message.
        """
        agent_id = message.content.get("agent_id")
        
        if agent_id in self.agent_registry:
            logger.info(f"Stopping agent {agent_id}")
            
            # TODO: Implement actual agent stopping logic
            # This would involve stopping the agent process or thread
            
            # Update status in registry
            self.agent_registry[agent_id]["status"] = AgentStatus.STOPPED.value
            
            # Send confirmation
            response = Message(
                msg_type=MessageType.RESPONSE,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "status": "success",
                    "message": f"Agent {agent_id} stopped successfully"
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
        else:
            logger.warning(f"Attempted to stop unknown agent: {agent_id}")
            
            # Send error
            response = Message(
                msg_type=MessageType.ERROR,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "error": f"Unknown agent: {agent_id}",
                    "original_message_id": message.id
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
    
    async def _create_workflow(self, message: Message) -> None:
        """
        Create a new workflow for coordinating multi-step agent tasks.
        
        Args:
            message: The create workflow command message.
        """
        workflow_data = message.content.get("workflow_data", {})
        workflow_id = workflow_data.get("id", str(uuid.uuid4()))
        
        if not workflow_data.get("steps"):
            logger.error("Invalid workflow data: missing steps")
            
            # Send error
            response = Message(
                msg_type=MessageType.ERROR,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "error": "Invalid workflow data: missing steps",
                    "original_message_id": message.id
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
            return
        
        # Register the workflow
        self.workflow_registry[workflow_id] = {
            "id": workflow_id,
            "name": workflow_data.get("name", f"Workflow-{workflow_id}"),
            "steps": workflow_data.get("steps", []),
            "status": "created",
            "current_step": 0,
            "results": {},
            "created_at": self.last_active,
            "updated_at": self.last_active,
            "creator_id": message.sender_id
        }
        
        logger.info(f"Created workflow {workflow_id}")
        
        # Send confirmation
        response = Message(
            msg_type=MessageType.RESPONSE,
            sender_id=self.id,
            target_id=message.sender_id,
            content={
                "status": "success",
                "message": f"Workflow {workflow_id} created successfully",
                "workflow_id": workflow_id
            },
            correlation_id=message.correlation_id
        )
        await self.send_message(response)
    
    async def _handle_agent_status_query(self, message: Message) -> None:
        """
        Handle a query for agent status.
        
        Args:
            message: The agent status query message.
        """
        agent_id = message.content.get("agent_id")
        
        if agent_id in self.agent_registry:
            # Get agent status
            agent_status = self.agent_registry[agent_id]
            
            # Send response
            response = Message(
                msg_type=MessageType.RESPONSE,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "agent_status": agent_status
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
        else:
            logger.warning(f"Status query for unknown agent: {agent_id}")
            
            # Send error
            response = Message(
                msg_type=MessageType.ERROR,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "error": f"Unknown agent: {agent_id}",
                    "original_message_id": message.id
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
    
    async def _handle_agents_by_type_query(self, message: Message) -> None:
        """
        Handle a query for agents by type.
        
        Args:
            message: The agents by type query message.
        """
        agent_type = message.content.get("agent_type")
        
        if agent_type in self.agent_type_registry:
            # Get agents of the specified type
            agent_ids = list(self.agent_type_registry[agent_type])
            agents = {agent_id: self.agent_registry[agent_id] for agent_id in agent_ids if agent_id in self.agent_registry}
            
            # Send response
            response = Message(
                msg_type=MessageType.RESPONSE,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "agent_type": agent_type,
                    "agents": agents
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
        else:
            logger.warning(f"Query for unknown agent type: {agent_type}")
            
            # Send response with empty list
            response = Message(
                msg_type=MessageType.RESPONSE,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "agent_type": agent_type,
                    "agents": {}
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
    
    async def _handle_system_status_query(self, message: Message) -> None:
        """
        Handle a query for overall system status.
        
        Args:
            message: The system status query message.
        """
        # Compile system status
        system_status = {
            "total_agents": len(self.agent_registry),
            "agent_types": {
                agent_type: len(agents) 
                for agent_type, agents in self.agent_type_registry.items()
            },
            "active_workflows": len(self.workflow_registry),
            "active_tasks": len(self.task_registry),
            "orchestrator_status": self.get_status()
        }
        
        # Send response
        response = Message(
            msg_type=MessageType.RESPONSE,
            sender_id=self.id,
            target_id=message.sender_id,
            content={
                "system_status": system_status
            },
            correlation_id=message.correlation_id
        )
        await self.send_message(response)
    
    async def _handle_task_status_query(self, message: Message) -> None:
        """
        Handle a query for task status.
        
        Args:
            message: The task status query message.
        """
        task_id = message.content.get("task_id")
        
        if task_id in self.task_registry:
            # Get task status
            task_status = self.task_registry[task_id]
            
            # Send response
            response = Message(
                msg_type=MessageType.RESPONSE,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "task_status": task_status
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
        else:
            logger.warning(f"Status query for unknown task: {task_id}")
            
            # Send error
            response = Message(
                msg_type=MessageType.ERROR,
                sender_id=self.id,
                target_id=message.sender_id,
                content={
                    "error": f"Unknown task: {task_id}",
                    "original_message_id": message.id
                },
                correlation_id=message.correlation_id
            )
            await self.send_message(response)
    
    async def _process_outbound_messages(self) -> None:
        """Process outbound messages from the outbound queue."""
        while self.status != AgentStatus.STOPPED:
            try:
                # Wait for the next message
                message = await self.outbound_queue.get()
                
                # Process the message
                # In a real implementation, this would involve sending
                # the message to its target through some communication channel
                logger.debug(f"Processing outbound message {message.id} to {message.target_id}")
                
                # For now, we just log it
                logger.info(f"Would send message from {message.sender_id} to {message.target_id}: {message.type.value}")
                
                # Mark the task as done
                self.outbound_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info(f"Outbound message processing for agent {self.id} was cancelled")
                break
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error processing outbound message: {str(e)}")
    
    async def register_and_start_agents(self, agents: List[Agent]) -> None:
        """
        Register and start multiple agents.
        
        Args:
            agents: The list of agents to register and start.
        """
        for agent in agents:
            # Register the agent
            self.agent_registry[agent.id] = agent.get_status()
            
            # Add to type registry
            if agent.type not in self.agent_type_registry:
                self.agent_type_registry[agent.type] = set()
            self.agent_type_registry[agent.type].add(agent.id)
            
            # Start the agent
            try:
                await agent.start()
                logger.info(f"Started agent {agent.id}")
            except Exception as e:
                logger.error(f"Failed to start agent {agent.id}: {str(e)}")
    
    async def stop_all_agents(self) -> None:
        """Stop all registered agents."""
        for agent_id, agent_data in self.agent_registry.items():
            if agent_data.get("status") == AgentStatus.RUNNING.value:
                # TODO: Implement actual agent stopping logic
                # This would involve stopping the agent process or thread
                
                # Update status in registry
                self.agent_registry[agent_id]["status"] = AgentStatus.STOPPED.value
                
                logger.info(f"Stopped agent {agent_id}")
    
    async def _cleanup(self) -> None:
        """Clean up resources when the orchestrator is stopping."""
        # Stop all agents
        await self.stop_all_agents()
        
        # Cancel any ongoing tasks
        # (In a real implementation, this might involve more complex cleanup)
        
        logger.info("Orchestrator cleanup completed")
