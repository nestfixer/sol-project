"""
Main entry point for the Solana Token Analysis Agent Swarm.
Initializes the system, creates agents, and provides a command-line interface.
"""

import argparse
import asyncio
import os
import sys
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from config.config_manager import ConfigManager
from core.knowledge_base import KnowledgeBase
from core.orchestrator_agent import OrchestratorAgent
from agents.data_collection_agent import DataCollectionAgent
from agents.pattern_analysis_agent import PatternAnalysisAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from agents.investment_advisory_agent import InvestmentAdvisoryAgent


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging based on configuration.
    
    Args:
        config: Logging configuration.
    """
    log_level = config.get("level", "INFO")
    log_format = config.get("format", "{time} | {level} | {message}")
    log_file = config.get("file_path")
    
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stderr,
        level=log_level,
        format=log_format
    )
    
    # Add file logger if configured
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        logger.add(
            log_file,
            level=log_level,
            format=log_format,
            rotation=config.get("rotation", "1 day"),
            retention=config.get("retention", "1 week")
        )
    
    logger.info(f"Logging initialized at level {log_level}")


async def run_agent_swarm(config_path: str) -> None:
    """
    Initialize and run the agent swarm.
    
    Args:
        config_path: Path to the configuration file.
    """
    try:
        # Load configuration
        config_manager = ConfigManager(config_path)
        config = config_manager.load_config()
        
        # Set up logging
        setup_logging(config.logging.dict())
        
        logger.info(f"Starting {config.name} v{config.version}")
        
        # Create data directories if they don't exist
        kb_path = config.knowledge_base_path
        if kb_path:
            os.makedirs(os.path.dirname(os.path.abspath(kb_path)), exist_ok=True)
        
        # Create knowledge base
        knowledge_base = KnowledgeBase(persistence_path=kb_path)
        logger.info(f"Knowledge base initialized with persistence path: {kb_path}")
        
        # Create orchestrator agent
        orchestrator = OrchestratorAgent()
        await orchestrator.start()
        logger.info(f"Orchestrator agent started: {orchestrator.id}")
        
        # Create specialized agents
        agents = []
        
        # Data Collection Agent
        data_collector_config = config.agents.get("data_collection", {}).dict()
        if data_collector_config.get("enabled", True):
            for i in range(data_collector_config.get("count", 1)):
                data_collector = DataCollectionAgent(
                    rpc_url=config.solana.rpc_url,
                    websocket_url=config.solana.websocket_url,
                    knowledge_base=knowledge_base
                )
                agents.append(data_collector)
                logger.info(f"Data Collection Agent created: {data_collector.id}")
        
        # Pattern Analysis Agent
        pattern_analyzer_config = config.agents.get("pattern_analysis", {}).dict()
        if pattern_analyzer_config.get("enabled", True):
            for i in range(pattern_analyzer_config.get("count", 1)):
                pattern_analyzer = PatternAnalysisAgent(
                    knowledge_base=knowledge_base,
                    config=pattern_analyzer_config.get("params", {})
                )
                agents.append(pattern_analyzer)
                logger.info(f"Pattern Analysis Agent created: {pattern_analyzer.id}")
        
        # Risk Assessment Agent
        risk_assessor_config = config.agents.get("risk_assessment", {}).dict()
        if risk_assessor_config.get("enabled", True):
            for i in range(risk_assessor_config.get("count", 1)):
                risk_assessor = RiskAssessmentAgent(
                    knowledge_base=knowledge_base,
                    config=risk_assessor_config.get("params", {})
                )
                agents.append(risk_assessor)
                logger.info(f"Risk Assessment Agent created: {risk_assessor.id}")
        
        # Investment Advisory Agent
        investment_advisor_config = config.agents.get("investment_advisory", {}).dict()
        if investment_advisor_config.get("enabled", True):
            for i in range(investment_advisor_config.get("count", 1)):
                investment_advisor = InvestmentAdvisoryAgent(
                    knowledge_base=knowledge_base,
                    config=investment_advisor_config.get("params", {})
                )
                agents.append(investment_advisor)
                logger.info(f"Investment Advisory Agent created: {investment_advisor.id}")
        
        # Register and start all agents with the orchestrator
        await orchestrator.register_and_start_agents(agents)
        logger.info(f"Started {len(agents)} agents")
        
        # Set dependencies between agents
        for agent in agents:
            if isinstance(agent, InvestmentAdvisoryAgent):
                # Find pattern analysis and risk assessment agents for dependencies
                pattern_analyzer_id = next(
                    (a.id for a in agents if isinstance(a, PatternAnalysisAgent)),
                    None
                )
                risk_assessor_id = next(
                    (a.id for a in agents if isinstance(a, RiskAssessmentAgent)),
                    None
                )
                
                if pattern_analyzer_id and risk_assessor_id:
                    # Create a message to set dependencies
                    from core.agent_base import Message, MessageType
                    msg = Message(
                        msg_type=MessageType.COMMAND,
                        sender_id=orchestrator.id,
                        target_id=agent.id,
                        content={
                            "command": "set_dependencies",
                            "pattern_analysis_agent_id": pattern_analyzer_id,
                            "risk_assessment_agent_id": risk_assessor_id
                        }
                    )
                    await agent.receive_message(msg)
                    logger.info(f"Set dependencies for Investment Advisory Agent {agent.id}")
        
        logger.info("Agent swarm is now running")
        
        # Keep the program running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Shutting down agent swarm")
            await orchestrator.stop()
            logger.info("Agent swarm shutdown complete")
    
    except Exception as e:
        logger.error(f"Error running agent swarm: {str(e)}")
        raise


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Solana Token Analysis Agent Swarm")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--create-config", action="store_true", help="Create default config file")
    return parser.parse_args()


def main():
    """Main entry point for the agent swarm."""
    args = parse_args()
    
    # Create default config if requested
    if args.create_config:
        config_manager = ConfigManager(args.config)
        config_manager.create_default_config()
        print(f"Created default configuration at {args.config}")
        return
    
    # Run the agent swarm
    asyncio.run(run_agent_swarm(args.config))


if __name__ == "__main__":
    main()
