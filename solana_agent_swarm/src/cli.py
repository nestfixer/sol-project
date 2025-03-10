"""
Command-line interface for interacting with the Solana Token Analysis Agent Swarm.
Provides a simple way to test and use the system.
"""

import argparse
import asyncio
import cmd
import json
import os
import sys
from typing import Any, Dict, List, Optional

from loguru import logger

from config.config_manager import ConfigManager
from core.knowledge_base import KnowledgeBase, EntryType
from core.orchestrator_agent import OrchestratorAgent
from core.agent_base import Message, MessageType
from agents.data_collection_agent import DataCollectionAgent
from agents.pattern_analysis_agent import PatternAnalysisAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from agents.investment_advisory_agent import InvestmentAdvisoryAgent


class SolanaTokenAnalysisCLI(cmd.Cmd):
    """Interactive CLI for the Solana Token Analysis Agent Swarm."""
    
    intro = """
╔═══════════════════════════════════════════════════╗
║ Solana Token Analysis Agent Swarm CLI             ║
║ Type 'help' or '?' to list commands.              ║
║ Type 'exit' or 'quit' to exit.                    ║
╚═══════════════════════════════════════════════════╝
"""
    prompt = "solana-swarm> "
    
    def __init__(self, config_path: str):
        """
        Initialize the CLI.
        
        Args:
            config_path: Path to the configuration file.
        """
        super().__init__()
        self.config_path = config_path
        self.orchestrator = None
        self.knowledge_base = None
        self.agents = {}  # agent_type -> agent
        self.agent_map = {}  # agent_id -> agent
        self.running = False
        self.response_futures = {}  # correlation_id -> future
        
        # Load configuration
        try:
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.load_config()
            print(f"✅ Loaded configuration from {config_path}")
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {config_path}")
            print(f"Run 'python cli.py --create-config' to create a default configuration.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading configuration: {str(e)}")
            sys.exit(1)
    
    async def initialize_system(self) -> None:
        """Initialize the agent swarm."""
        try:
            # Create knowledge base
            kb_path = self.config.knowledge_base_path
            if kb_path:
                os.makedirs(os.path.dirname(os.path.abspath(kb_path)), exist_ok=True)
            
            self.knowledge_base = KnowledgeBase(persistence_path=kb_path)
            print(f"✅ Knowledge base initialized")
            
            # Create orchestrator agent
            self.orchestrator = OrchestratorAgent()
            await self.orchestrator.start()
            print(f"✅ Orchestrator agent started: {self.orchestrator.id}")
            
            # Create specialized agents
            # Data Collection Agent
            data_collector_config = self.config.agents.get("data_collection", {}).dict()
            if data_collector_config.get("enabled", True):
                data_collector = DataCollectionAgent(
                    rpc_url=self.config.solana.rpc_url,
                    websocket_url=self.config.solana.websocket_url,
                    knowledge_base=self.knowledge_base
                )
                self.agents["data_collection"] = data_collector
                self.agent_map[data_collector.id] = data_collector
                print(f"✅ Data Collection Agent created: {data_collector.id}")
            
            # Pattern Analysis Agent
            pattern_analyzer_config = self.config.agents.get("pattern_analysis", {}).dict()
            if pattern_analyzer_config.get("enabled", True):
                pattern_analyzer = PatternAnalysisAgent(
                    knowledge_base=self.knowledge_base,
                    config=pattern_analyzer_config.get("params", {})
                )
                self.agents["pattern_analysis"] = pattern_analyzer
                self.agent_map[pattern_analyzer.id] = pattern_analyzer
                print(f"✅ Pattern Analysis Agent created: {pattern_analyzer.id}")
            
            # Risk Assessment Agent
            risk_assessor_config = self.config.agents.get("risk_assessment", {}).dict()
            if risk_assessor_config.get("enabled", True):
                risk_assessor = RiskAssessmentAgent(
                    knowledge_base=self.knowledge_base,
                    config=risk_assessor_config.get("params", {})
                )
                self.agents["risk_assessment"] = risk_assessor
                self.agent_map[risk_assessor.id] = risk_assessor
                print(f"✅ Risk Assessment Agent created: {risk_assessor.id}")
            
            # Investment Advisory Agent
            investment_advisor_config = self.config.agents.get("investment_advisory", {}).dict()
            if investment_advisor_config.get("enabled", True):
                investment_advisor = InvestmentAdvisoryAgent(
                    knowledge_base=self.knowledge_base,
                    config=investment_advisor_config.get("params", {})
                )
                self.agents["investment_advisory"] = investment_advisor
                self.agent_map[investment_advisor.id] = investment_advisor
                print(f"✅ Investment Advisory Agent created: {investment_advisor.id}")
            
            # Register all agents with the orchestrator
            await self.orchestrator.register_and_start_agents(list(self.agent_map.values()))
            print(f"✅ Started {len(self.agent_map)} agents")
            
            # Set dependencies between agents
            if "investment_advisory" in self.agents and "pattern_analysis" in self.agents and "risk_assessment" in self.agents:
                inv_advisor = self.agents["investment_advisory"]
                pattern_analyzer = self.agents["pattern_analysis"]
                risk_assessor = self.agents["risk_assessment"]
                
                # Create a message to set dependencies
                msg = Message(
                    msg_type=MessageType.COMMAND,
                    sender_id=self.orchestrator.id,
                    target_id=inv_advisor.id,
                    content={
                        "command": "set_dependencies",
                        "pattern_analysis_agent_id": pattern_analyzer.id,
                        "risk_assessment_agent_id": risk_assessor.id
                    }
                )
                await inv_advisor.receive_message(msg)
                print(f"✅ Set dependencies for Investment Advisory Agent")
            
            # Start the response handler
            self.running = True
            asyncio.create_task(self.handle_responses())
            
            print("✅ System initialized and ready")
        
        except Exception as e:
            print(f"❌ Error initializing system: {str(e)}")
            raise
    
    async def handle_responses(self) -> None:
        """Handle responses from agents."""
        while self.running:
            # Process any completed futures
            completed_ids = []
            
            for correlation_id, future in self.response_futures.items():
                if future.done():
                    try:
                        result = future.result()
                        print(f"\n--- Response received ---")
                        print(json.dumps(result, indent=2))
                        print(f"------------------------\n")
                        print(self.prompt, end="", flush=True)
                    except Exception as e:
                        print(f"\n❌ Error processing response: {str(e)}\n")
                        print(self.prompt, end="", flush=True)
                    
                    completed_ids.append(correlation_id)
            
            # Remove completed futures
            for correlation_id in completed_ids:
                del self.response_futures[correlation_id]
            
            # Sleep briefly
            await asyncio.sleep(0.1)
    
    async def cleanup(self) -> None:
        """Clean up resources when shutting down."""
        self.running = False
        
        if self.orchestrator:
            await self.orchestrator.stop()
            print("✅ Orchestrator stopped")
        
        print("✅ System shutdown complete")
    
    async def send_message_and_wait(self, message: Message) -> Dict[str, Any]:
        """
        Send a message and wait for the response.
        
        Args:
            message: The message to send.
            
        Returns:
            The response content.
        """
        # Create a future for the response
        future = asyncio.Future()
        self.response_futures[message.correlation_id] = future
        
        # Add a handler for this response
        target_agent = self.agent_map.get(message.target_id)
        if target_agent:
            await target_agent.receive_message(message)
            print(f"📤 Sent message to {target_agent.type} agent")
        else:
            print(f"❌ Unknown agent ID: {message.target_id}")
            future.set_result({"error": f"Unknown agent ID: {message.target_id}"})
        
        return await future
    
    def do_start(self, arg):
        """Start the agent swarm."""
        asyncio.create_task(self.initialize_system())
    
    def do_collect_token_data(self, arg):
        """
        Collect data for a token.
        Usage: collect_token_data <token_address>
        """
        parts = arg.split()
        if not parts:
            print("❌ Token address required")
            return
        
        token_address = parts[0]
        force_refresh = "--force" in parts
        
        if "data_collection" not in self.agents:
            print("❌ Data Collection Agent not available")
            return
        
        data_collector = self.agents["data_collection"]
        
        async def collect_token_data():
            # Create a message to collect token data
            correlation_id = f"collect-{token_address}-{str(uuid.uuid4())[:8]}"
            message = Message(
                msg_type=MessageType.COMMAND,
                sender_id="cli",
                target_id=data_collector.id,
                content={
                    "command": "collect_token_data",
                    "token_address": token_address,
                    "force_refresh": force_refresh
                },
                correlation_id=correlation_id
            )
            
            # Send the message and wait for the response
            await self.send_message_and_wait(message)
        
        asyncio.create_task(collect_token_data())
    
    def do_analyze_patterns(self, arg):
        """
        Analyze patterns for a token.
        Usage: analyze_patterns <token_address>
        """
        parts = arg.split()
        if not parts:
            print("❌ Token address required")
            return
        
        token_address = parts[0]
        force_refresh = "--force" in parts
        
        if "pattern_analysis" not in self.agents:
            print("❌ Pattern Analysis Agent not available")
            return
        
        pattern_analyzer = self.agents["pattern_analysis"]
        
        async def analyze_patterns():
            # Create a message to analyze patterns
            correlation_id = f"patterns-{token_address}-{str(uuid.uuid4())[:8]}"
            message = Message(
                msg_type=MessageType.COMMAND,
                sender_id="cli",
                target_id=pattern_analyzer.id,
                content={
                    "command": "analyze_token",
                    "token_address": token_address,
                    "force_refresh": force_refresh
                },
                correlation_id=correlation_id
            )
            
            # Send the message and wait for the response
            await self.send_message_and_wait(message)
        
        asyncio.create_task(analyze_patterns())
    
    def do_assess_risk(self, arg):
        """
        Assess risk for a token.
        Usage: assess_risk <token_address>
        """
        parts = arg.split()
        if not parts:
            print("❌ Token address required")
            return
        
        token_address = parts[0]
        force_refresh = "--force" in parts
        
        if "risk_assessment" not in self.agents:
            print("❌ Risk Assessment Agent not available")
            return
        
        risk_assessor = self.agents["risk_assessment"]
        
        async def assess_risk():
            # Create a message to assess risk
            correlation_id = f"risk-{token_address}-{str(uuid.uuid4())[:8]}"
            message = Message(
                msg_type=MessageType.COMMAND,
                sender_id="cli",
                target_id=risk_assessor.id,
                content={
                    "command": "assess_token_risk",
                    "token_address": token_address,
                    "force_refresh": force_refresh
                },
                correlation_id=correlation_id
            )
            
            # Send the message and wait for the response
            await self.send_message_and_wait(message)
        
        asyncio.create_task(assess_risk())
    
    def do_evaluate_opportunity(self, arg):
        """
        Evaluate investment opportunity for a token.
        Usage: evaluate_opportunity <token_address>
        """
        parts = arg.split()
        if not parts:
            print("❌ Token address required")
            return
        
        token_address = parts[0]
        force_refresh = "--force" in parts
        
        if "investment_advisory" not in self.agents:
            print("❌ Investment Advisory Agent not available")
            return
        
        investment_advisor = self.agents["investment_advisory"]
        
        async def evaluate_opportunity():
            # Create a message to evaluate opportunity
            correlation_id = f"opportunity-{token_address}-{str(uuid.uuid4())[:8]}"
            message = Message(
                msg_type=MessageType.COMMAND,
                sender_id="cli",
                target_id=investment_advisor.id,
                content={
                    "command": "evaluate_opportunity",
                    "token_address": token_address,
                    "force_refresh": force_refresh
                },
                correlation_id=correlation_id
            )
            
            # Send the message and wait for the response
            await self.send_message_and_wait(message)
        
        asyncio.create_task(evaluate_opportunity())
    
    def do_get_top_opportunities(self, arg):
        """
        Get top investment opportunities.
        Usage: get_top_opportunities [count]
        """
        try:
            count = int(arg) if arg else 5
        except ValueError:
            print("❌ Invalid count, using default (5)")
            count = 5
        
        if "investment_advisory" not in self.agents:
            print("❌ Investment Advisory Agent not available")
            return
        
        investment_advisor = self.agents["investment_advisory"]
        
        async def get_top_opportunities():
            # Create a message to get top opportunities
            correlation_id = f"top-opportunities-{str(uuid.uuid4())[:8]}"
            message = Message(
                msg_type=MessageType.QUERY,
                sender_id="cli",
                target_id=investment_advisor.id,
                content={
                    "query_type": "get_top_opportunities",
                    "count": count,
                    "min_confidence": "medium"
                },
                correlation_id=correlation_id
            )
            
            # Send the message and wait for the response
            await self.send_message_and_wait(message)
        
        asyncio.create_task(get_top_opportunities())
    
    def do_system_status(self, arg):
        """Show system status."""
        if not self.orchestrator:
            print("❌ System not initialized")
            return
        
        async def get_system_status():
            # Create a message to get system status
            correlation_id = f"system-status-{str(uuid.uuid4())[:8]}"
            message = Message(
                msg_type=MessageType.QUERY,
                sender_id="cli",
                target_id=self.orchestrator.id,
                content={
                    "query_type": "system_status"
                },
                correlation_id=correlation_id
            )
            
            # Send the message and wait for the response
            await self.send_message_and_wait(message)
        
        asyncio.create_task(get_system_status())
    
    def do_exit(self, arg):
        """Exit the CLI."""
        print("Shutting down agent swarm...")
        asyncio.create_task(self.cleanup())
        return True
    
    def do_quit(self, arg):
        """Exit the CLI."""
        return self.do_exit(arg)
    
    def default(self, line):
        """Handle unknown commands."""
        print(f"Unknown command: {line}")
        print("Type 'help' or '?' to list available commands.")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Solana Token Analysis CLI")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--create-config", action="store_true", help="Create default config file")
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Create default config if requested
    if args.create_config:
        config_manager = ConfigManager(args.config)
        config_manager.create_default_config()
        print(f"Created default configuration at {args.config}")
        return
    
    # Create the CLI
    cli = SolanaTokenAnalysisCLI(args.config)
    
    # Run the CLI in asyncio event loop
    loop = asyncio.get_event_loop()
    
    try:
        loop.run_until_complete(asyncio.gather(
            loop.run_in_executor(None, cli.cmdloop)
        ))
    except KeyboardInterrupt:
        print("\nShutting down...")
        loop.run_until_complete(cli.cleanup())
    finally:
        loop.close()


if __name__ == "__main__":
    import uuid  # Import here to avoid linting errors
    main()
