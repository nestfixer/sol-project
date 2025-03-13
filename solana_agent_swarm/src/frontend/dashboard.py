"""
Solana Token Analysis Agent Swarm Dashboard

This module provides both a Streamlit-based dashboard and a React-based frontend
for visualizing and interacting with the Solana Token Analysis Agent Swarm.
"""

import asyncio
import json
import datetime
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Import the bridge to MCP servers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from mcp_integration import SolanaAgentKitBridge
except ImportError:
    print("Error: Could not import SolanaAgentKitBridge")
    sys.exit(1)

# React frontend paths
FRONTEND_DIR = os.path.join(current_dir)

# Helper function to run async functions
def run_async(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

# Initialize the bridge (shared between both interfaces)
def get_bridge():
    return SolanaAgentKitBridge()

#######################################
# React Flask Server Implementation
#######################################

def create_flask_app():
    """Create and configure the Flask app for the React frontend."""
    try:
        from flask import Flask, jsonify, request, send_from_directory
        from flask_cors import CORS
    except ImportError:
        print("Flask is not installed. Install with: pip install flask flask-cors")
        sys.exit(1)
        
    app = Flask(__name__, static_folder=FRONTEND_DIR)
    CORS(app)  # Enable CORS for all routes
    bridge = get_bridge()
    
    # Serve React static files
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path == "":
            return send_from_directory(FRONTEND_DIR, 'index.html')
        try:
            return send_from_directory(FRONTEND_DIR, path)
        except:
            return send_from_directory(FRONTEND_DIR, 'index.html')
    
    # API endpoints for React frontend
    @app.route('/api/token/<token_address>', methods=['GET'])
    def get_token_info(token_address):
        try:
            token_info = run_async(bridge.get_token_info(token_address))
            price_data = run_async(bridge.get_token_price(token_address))
            
            return jsonify({
                "info": token_info,
                "price": price_data,
                "timestamp": datetime.datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/patterns/<token_address>', methods=['GET'])
    def get_token_patterns(token_address):
        time_period = request.args.get('time_period', '7d')
        try:
            pattern_data = run_async(bridge.analyze_token_pattern(token_address, time_period))
            return jsonify(pattern_data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/market_patterns', methods=['GET'])
    def get_market_patterns():
        pattern_type = request.args.get('pattern_type', 'all')
        min_confidence = float(request.args.get('min_confidence', 0.7))
        try:
            market_data = run_async(bridge.detect_market_patterns(pattern_type, min_confidence))
            return jsonify(market_data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/price_prediction/<token_address>', methods=['GET'])
    def get_price_prediction(token_address):
        time_horizon = request.args.get('time_horizon', '24h')
        try:
            prediction = run_async(bridge.get_price_prediction(token_address, time_horizon))
            return jsonify(prediction)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/risk/<token_address>', methods=['GET'])
    def assess_token_risk(token_address):
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        try:
            risk_data = run_async(bridge.assess_token_risk(token_address, force_refresh))
            return jsonify(risk_data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/wallet_tokens/<wallet_address>', methods=['GET'])
    def get_wallet_tokens(wallet_address):
        try:
            wallet_tokens = run_async(bridge.get_wallet_tokens(wallet_address))
            return jsonify(wallet_tokens)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/trending_tokens', methods=['GET'])
    def get_trending_tokens():
        try:
            trending_data = run_async(bridge.get_trending_tokens())
            return jsonify(trending_data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    return app

def run_react_server(host='127.0.0.1', port=8080):
    """Run the React frontend server."""
    app = create_flask_app()
    print(f"Starting React frontend server at http://{host}:{port}")
    print(f"Press Ctrl+C to quit")
    app.run(host=host, port=port, debug=True)

#######################################
# Streamlit Implementation
#######################################

def run_streamlit_app():
    """
    Main Streamlit application - this is a separate function
    to ensure set_page_config is always first.
    """
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    
    # This must be the first Streamlit command
    st.set_page_config(
        page_title="Solana Token Analysis Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize bridge
    bridge = get_bridge()
    
    # Initialize session state
    if "token_data" not in st.session_state:
        st.session_state.token_data = {}
    if "pattern_data" not in st.session_state:
        st.session_state.pattern_data = {}
    if "risk_data" not in st.session_state:
        st.session_state.risk_data = {}
    if "blacklist" not in st.session_state:
        st.session_state.blacklist = {"wallets": [], "tokens": []}
    if "trending_tokens" not in st.session_state:
        st.session_state.trending_tokens = []
    if "market_patterns" not in st.session_state:
        st.session_state.market_patterns = []
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Token Analysis"
    
    # Sidebar
    st.sidebar.title("Solana Token Analysis")
    st.sidebar.image("https://solana.com/src/img/branding/solanaLogoMark.svg", width=100)
    
    # Navigation tabs
    tabs = ["Token Analysis", "Pattern Detection", "Risk Assessment", "Wallet Tracking"]
    st.session_state.active_tab = st.sidebar.radio("Navigation", tabs, index=tabs.index(st.session_state.active_tab))
    
    # Token address input
    token_address = st.sidebar.text_input(
        "Token Address", 
        value="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # Default to USDC
        help="Enter the Solana token address (mint) to analyze"
    )
    
    st.sidebar.divider()
    
    # Trending tokens
    if st.sidebar.button("Get Trending Tokens"):
        with st.sidebar.status("Fetching trending tokens..."):
            try:
                trending_data = run_async(bridge.get_trending_tokens())
                if trending_data and "tokens" in trending_data:
                    st.session_state.trending_tokens = trending_data["tokens"]
                    st.sidebar.success(f"Found {len(st.session_state.trending_tokens)} trending tokens")
                else:
                    st.sidebar.warning("No trending tokens found")
            except Exception as e:
                st.sidebar.error(f"Error fetching trending tokens: {str(e)}")
    
    if st.session_state.trending_tokens:
        st.sidebar.subheader("Trending Tokens")
        for token in st.session_state.trending_tokens[:5]:  # Show top 5
            if st.sidebar.button(f"{token['symbol']} - {token['name']}", key=f"trending_{token['address']}"):
                token_address = token['address']
                st.experimental_rerun()
    
    # Main content area
    if st.session_state.active_tab == "Token Analysis":
        st.title("Token Analysis Dashboard")
        
        if token_address:
            # Fetch token data if not in session state
            if token_address not in st.session_state.token_data:
                with st.status("Fetching token data..."):
                    try:
                        token_info = run_async(bridge.get_token_info(token_address))
                        price_data = run_async(bridge.get_token_price(token_address))
                        
                        st.session_state.token_data[token_address] = {
                            "info": token_info,
                            "price": price_data,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    except Exception as e:
                        st.error(f"Error fetching token data: {str(e)}")
            
            # Display token data
            if token_address in st.session_state.token_data:
                token_data = st.session_state.token_data[token_address]
                token_info = token_data["info"]
                
                # Token overview in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Token Overview")
                    st.write(f"**Name:** {token_info.get('name', 'Unknown')}")
                    st.write(f"**Symbol:** {token_info.get('symbol', 'Unknown')}")
                    st.write(f"**Address:** {token_address}")
                    st.write(f"**Decimals:** {token_info.get('decimals', 'Unknown')}")
                
                with col2:
                    st.subheader("Market Data")
                    market = token_info.get('market', {})
                    st.write(f"**Price:** ${market.get('price_usd', 'Unknown')}")
                    st.write(f"**24h Change:** {market.get('price_change_24h', 'Unknown')}%")
                    st.write(f"**24h Volume:** ${market.get('volume_24h', 'Unknown')}")
                    st.write(f"**Market Cap:** ${market.get('market_cap', 'Unknown')}")
                
                with col3:
                    st.subheader("Holder Information")
                    holders = token_info.get('holders', {})
                    st.write(f"**Total Holders:** {holders.get('count', 'Unknown')}")
                    
                    if 'distribution' in holders and holders['distribution']:
                        dist_data = holders['distribution']
                        labels = [f"Wallet {i+1}" for i in range(len(dist_data))]
                        values = [d['percentage'] for d in dist_data]
                        
                        if sum(values) < 100:
                            labels.append("Others")
                            values.append(100 - sum(values))
                        
                        fig = px.pie(values=values, names=labels, title="Token Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Refresh button
                if st.button("Refresh Token Data"):
                    if token_address in st.session_state.token_data:
                        del st.session_state.token_data[token_address]
                    st.experimental_rerun()
                
                # Price chart (simulated)
                st.subheader("Price History")
                st.write("*Note: This is simulated price data for demonstration purposes*")
                
                # Generate simulated data
                days = 30
                dates = [datetime.datetime.now() - datetime.timedelta(days=i) for i in range(days)]
                dates.reverse()
                
                # Create price chart
                base_price = float(market.get('price_usd', 1.0))
                amplitude = base_price * 0.2
                prices = [
                    base_price + amplitude * np.sin(i / 5) + amplitude * 0.5 * np.random.randn()
                    for i in range(days)
                ]
                
                # Ensure no negative prices
                prices = [max(0.0001, p) for p in prices]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, 
                    y=prices,
                    mode='lines',
                    name='Price',
                    line=dict(color='#9945FF', width=2)
                ))
                fig.update_layout(
                    title=f"{token_info.get('symbol', 'Token')} Price History (30 Days)",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif st.session_state.active_tab == "Pattern Detection":
        st.title("Pattern Detection Dashboard")
        st.write("Pattern detection functionality coming soon!")
    
    elif st.session_state.active_tab == "Risk Assessment":
        st.title("Risk Assessment Dashboard")
        st.write("Risk assessment functionality coming soon!")
    
    elif st.session_state.active_tab == "Wallet Tracking":
        st.title("Wallet Tracking Dashboard")
        st.write("Wallet tracking functionality coming soon!")
    
    # Footer
    st.markdown("---")
    st.caption("Solana Token Analysis Agent Swarm Dashboard | © 2025")

#######################################
# Main Entry Point
#######################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solana Token Analysis Agent Swarm Dashboard")
    parser.add_argument(
        "--react",
        action="store_true",
        help="Launch the React-based dashboard instead of Streamlit"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for the React server (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the React server (default: 8080)"
    )
    
    args = parser.parse_args()
    
    if args.react:
        # Launch React frontend
        run_react_server(host=args.host, port=args.port)
    else:
        # Check for Streamlit
        try:
            import streamlit as st
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("Error: Streamlit is not installed. Install with: pip install streamlit plotly")
            sys.exit(1)
        
        # Launch Streamlit interface - this is just a message
        # The actual app will be run by the Streamlit server when executing this script
        print("Starting Streamlit dashboard...")
        print("If Streamlit is not already running, execute:")
        print(f"  streamlit run {__file__}")
        
        # This is only reached when running directly with Python, not through Streamlit
        run_streamlit_app()
