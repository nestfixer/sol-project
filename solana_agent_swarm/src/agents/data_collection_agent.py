"""
Data Collection Agent for the Solana Token Analysis Agent Swarm.
Responsible for connecting to SolScan's RPC and WebSocket endpoints,
monitoring for new token events, and collecting token metadata.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

import aiohttp
import websockets
from loguru import logger

from ..core.agent_base import Agent, AgentStatus, Message, MessageType
from ..core.knowledge_base import EntryType, KnowledgeBase


class SolanaWebsocketClient:
    """Client for connecting to Solana WebSocket API."""
    
    def __init__(self, websocket_url: str, on_token_event):
        self.websocket_url = websocket_url
        self.on_token_event = on_token_event
        self.ws = None
        self.subscription_id = None
        self.is_connected = False
        self.keep_running = True
        self.reconnect_interval = 5  # seconds
        self.max_reconnect_attempts = 10
    
    async def connect(self):
        """Connect to the Solana WebSocket API."""
        try:
            self.ws = await websockets.connect(self.websocket_url)
            self.is_connected = True
            logger.info(f"Connected to WebSocket: {self.websocket_url}")
            
            # Subscribe to account notifications for token program
            # This is a simplified example - in practice, you would need to
            # filter by specific programs or set up more complex filters
            # based on your requirements
            TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
            
            subscribe_msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "programSubscribe",
                "params": [
                    TOKEN_PROGRAM_ID,
                    {
                        "encoding": "jsonParsed",
                        "commitment": "confirmed"
                    }
                ]
            }
            
            await self.ws.send(json.dumps(subscribe_msg))
            response = await self.ws.recv()
            response_data = json.loads(response)
            
            if "result" in response_data:
                self.subscription_id = response_data["result"]
                logger.info(f"Subscribed to program notifications with id: {self.subscription_id}")
            else:
                logger.error(f"Failed to subscribe: {response_data}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the Solana WebSocket API."""
        self.keep_running = False
        if self.ws and self.is_connected:
            # Unsubscribe if we have a subscription
            if self.subscription_id is not None:
                unsubscribe_msg = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "programUnsubscribe",
                    "params": [self.subscription_id]
                }
                try:
                    await self.ws.send(json.dumps(unsubscribe_msg))
                    response = await self.ws.recv()
                    logger.info(f"Unsubscribed: {response}")
                except Exception as e:
                    logger.warning(f"Error unsubscribing: {str(e)}")
            
            # Close the connection
            try:
                await self.ws.close()
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {str(e)}")
            
            self.is_connected = False
            self.subscription_id = None
    
    async def listen(self):
        """Listen for WebSocket messages."""
        reconnect_attempts = 0
        
        while self.keep_running:
            if not self.is_connected:
                # Try to reconnect
                if reconnect_attempts < self.max_reconnect_attempts:
                    logger.info(f"Attempting to reconnect (attempt {reconnect_attempts + 1}/{self.max_reconnect_attempts})...")
                    success = await self.connect()
                    if success:
                        reconnect_attempts = 0
                    else:
                        reconnect_attempts += 1
                        await asyncio.sleep(self.reconnect_interval)
                        continue
                else:
                    logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached. Giving up.")
                    break
            
            try:
                # Wait for messages
                response = await self.ws.recv()
                
                # Parse the message
                message = json.loads(response)
                
                # Check if it's a notification message
                if "method" in message and message["method"] == "programNotification":
                    # Extract relevant info from the notification
                    try:
                        params = message.get("params", {})
                        result = params.get("result", {})
                        
                        # Process token event - actual implementation would need to
                        # filter and extract specific token data based on your needs
                        if "value" in result:
                            await self.on_token_event(result)
                    except Exception as e:
                        logger.error(f"Error processing notification: {str(e)}")
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed unexpectedly")
                self.is_connected = False
                await asyncio.sleep(self.reconnect_interval)
            except Exception as e:
                logger.error(f"Error in WebSocket listener: {str(e)}")
                self.is_connected = False
                await asyncio.sleep(self.reconnect_interval)


class DataCollectionAgent(Agent):
    """
    Agent responsible for collecting token data from Solana blockchain.
    Connects to SolScan and other APIs to gather token metadata.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        websocket_url: Optional[str] = None,
        solscan_api_url: str = "https://api.solscan.io",
        knowledge_base: Optional[KnowledgeBase] = None,
    ):
        """
        Initialize the Data Collection Agent.
        
        Args:
            agent_id: Optional agent ID. If not provided, one will be generated.
            rpc_url: Solana RPC URL for API calls.
            websocket_url: Optional Solana WebSocket URL for real-time updates.
                If not provided, will be derived from RPC URL.
            solscan_api_url: SolScan API URL.
            knowledge_base: Optional knowledge base instance to use.
                If not provided, the agent will expect to receive one
                from the orchestrator after initialization.
        """
        agent_id = agent_id or f"data-collector-{str(uuid.uuid4())[:8]}"
        super().__init__(agent_id=agent_id, agent_type="DataCollectionAgent")
        
        self.rpc_url = rpc_url
        
        # Derive WebSocket URL from RPC URL if not provided
        if websocket_url is None and rpc_url.startswith("http"):
            websocket_url = rpc_url.replace("http", "ws")
        self.websocket_url = websocket_url
        
        self.solscan_api_url = solscan_api_url
        self.knowledge_base = knowledge_base
        
        # WebSocket client for real-time updates
        self.ws_client = None
        self.ws_task = None
        
        # HTTP session for API calls
        self.session = None
        
        # Token processing settings
        self.min_token_data_refresh_interval = 300  # seconds
        self.token_cache: Dict[str, Dict[str, Any]] = {}
        self.last_token_update: Dict[str, float] = {}
        
        logger.info(f"Data Collection Agent {self.id} initialized")
    
    async def _initialize(self) -> None:
        """Initialize the agent with API sessions and WebSocket connection."""
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        
        # Create and connect WebSocket client
        if self.websocket_url:
            self.ws_client = SolanaWebsocketClient(
                websocket_url=self.websocket_url,
                on_token_event=self._handle_token_event
            )
            connection_success = await self.ws_client.connect()
            if connection_success:
                # Start WebSocket listener task
                self.ws_task = asyncio.create_task(self.ws_client.listen())
                logger.info(f"Started WebSocket listener task for {self.id}")
            else:
                logger.error(f"Failed to establish WebSocket connection for {self.id}")
        
        logger.info(f"Data Collection Agent {self.id} ready")
    
    async def _handle_message(self, message: Message) -> None:
        """
        Handle incoming messages.
        
        Args:
            message: The message to handle.
        """
        logger.debug(f"Data Collection Agent {self.id} received message: {message.type.value}")
        
        if message.type == MessageType.COMMAND:
            await self._handle_command(message)
        elif message.type == MessageType.QUERY:
            await self._handle_query(message)
        else:
            logger.warning(f"Unsupported message type: {message.type.value}")
    
    async def _handle_command(self, message: Message) -> None:
        """
        Handle command messages.
        
        Args:
            message: The command message.
        """
        command = message.content.get("command")
        
        if command == "collect_token_data":
            # Handle command to collect data for a specific token
            token_address = message.content.get("token_address")
            force_refresh = message.content.get("force_refresh", False)
            
            if token_address:
                try:
                    token_data = await self._collect_token_data(token_address, force_refresh)
                    
                    # Send response with token data
                    response = Message(
                        msg_type=MessageType.RESPONSE,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "token_address": token_address,
                            "token_data": token_data
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
                            "error": f"Failed to collect token data: {str(e)}",
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
        
        if query_type == "get_token_info":
            # Handle query for token info
            token_address = message.content.get("token_address")
            
            if token_address:
                try:
                    token_data = await self._collect_token_data(token_address)
                    
                    # Send response with token data
                    response = Message(
                        msg_type=MessageType.RESPONSE,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "token_address": token_address,
                            "token_data": token_data
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
                            "error": f"Failed to get token info: {str(e)}",
                            "token_address": token_address
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(error_msg)
        
        elif query_type == "get_new_tokens":
            # Handle query for recently discovered tokens
            count = message.content.get("count", 10)
            
            try:
                # This would normally query our knowledge base for recent tokens
                # For this example, we'll just return a placeholder
                response = Message(
                    msg_type=MessageType.RESPONSE,
                    sender_id=self.id,
                    target_id=message.sender_id,
                    content={
                        "tokens": [
                            {"address": f"token{i}", "discovery_time": datetime.utcnow().isoformat()}
                            for i in range(count)
                        ]
                    },
                    correlation_id=message.correlation_id
                )
                await self.send_message(response)
            except Exception as e:
                error_msg = Message(
                    msg_type=MessageType.ERROR,
                    sender_id=self.id,
                    target_id=message.sender_id,
                    content={"error": f"Failed to get new tokens: {str(e)}"},
                    correlation_id=message.correlation_id
                )
                await self.send_message(error_msg)
        
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
    
    async def _handle_token_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle a token event from the WebSocket.
        
        Args:
            event_data: The event data from the WebSocket.
        """
        try:
            # Extract token information from the event
            # In a real implementation, this would involve parsing the specifics
            # of the event data structure
            logger.debug(f"Received token event: {event_data}")
            
            # Example implementation:
            if "pubkey" in event_data:
                token_address = event_data["pubkey"]
                logger.info(f"Detected token event for: {token_address}")
                
                # Collect token data
                token_data = await self._collect_token_data(token_address)
                
                # Store in knowledge base if available
                if self.knowledge_base is not None:
                    entry_id = f"token:{token_address}"
                    await self.knowledge_base.add_entry(
                        entry_id=entry_id,
                        entry_type=EntryType.TOKEN_DATA,
                        data=token_data,
                        metadata={
                            "discovery_event": "websocket",
                            "discovery_time": datetime.utcnow().isoformat()
                        },
                        source_agent_id=self.id,
                        tags=["token", "new_discovery"]
                    )
                    logger.info(f"Added token {token_address} to knowledge base")
                
                # Notify other agents about the new token
                event_msg = Message(
                    msg_type=MessageType.EVENT,
                    sender_id=self.id,
                    content={
                        "event_type": "new_token_discovered",
                        "token_address": token_address,
                        "discovery_time": datetime.utcnow().isoformat()
                    }
                )
                await self.send_message(event_msg)
            
        except Exception as e:
            logger.error(f"Error handling token event: {str(e)}")
    
    async def _collect_token_data(self, token_address: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Collect comprehensive data for a token.
        
        Args:
            token_address: The token's address.
            force_refresh: Whether to force a refresh of cached data.
            
        Returns:
            Token data including metadata, market info, and holders.
        """
        # Check cache first unless force refresh
        current_time = time.time()
        if (not force_refresh and 
            token_address in self.token_cache and
            token_address in self.last_token_update and
            current_time - self.last_token_update[token_address] < self.min_token_data_refresh_interval):
            return self.token_cache[token_address]
        
        # Fetch fresh data
        try:
            # In a real implementation, this would make API calls to SolScan, 
            # the Solana RPC, and possibly other services.
            # For this example, we'll construct a placeholder response.
            
            # Example SolScan API call for token info
            # token_info_url = f"{self.solscan_api_url}/token/meta?tokenAddress={token_address}"
            # async with self.session.get(token_info_url) as response:
            #     token_info = await response.json()
            
            # Placeholder data
            token_data = {
                "address": token_address,
                "name": f"Token {token_address[:6]}",
                "symbol": f"TKN{token_address[:3]}",
                "decimals": 9,
                "totalSupply": 1000000000,
                "market": {
                    "price_usd": 0.001,
                    "volume_24h": 10000,
                    "market_cap": 1000000,
                    "price_change_24h": 5.2
                },
                "holders": {
                    "count": 150,
                    "distribution": [
                        {"wallet": f"wallet1_{token_address[:4]}", "percentage": 15.5},
                        {"wallet": f"wallet2_{token_address[:4]}", "percentage": 10.2},
                        {"wallet": f"wallet3_{token_address[:4]}", "percentage": 8.7}
                    ]
                },
                "social": {
                    "twitter_followers": 1200,
                    "telegram_members": 850,
                    "sentiment_score": 0.65
                },
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Update cache
            self.token_cache[token_address] = token_data
            self.last_token_update[token_address] = current_time
            
            return token_data
        
        except Exception as e:
            logger.error(f"Error collecting token data for {token_address}: {str(e)}")
            raise
    
    async def _cleanup(self) -> None:
        """Clean up resources when stopping the agent."""
        # Close WebSocket connection
        if self.ws_client:
            await self.ws_client.disconnect()
        
        # Cancel WebSocket listener task
        if self.ws_task and not self.ws_task.done():
            self.ws_task.cancel()
            try:
                await self.ws_task
            except asyncio.CancelledError:
                pass
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        logger.info(f"Data Collection Agent {self.id} cleaned up")
