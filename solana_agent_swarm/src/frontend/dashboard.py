"""
Solana Token Analysis Agent Swarm Dashboard

This module provides a Streamlit-based dashboard for visualizing and interacting
with the Solana Token Analysis Agent Swarm.

Features:
1. Token Analysis Dashboard - Monitor token information and price data
2. Pattern Detection Dashboard - Visualize detected patterns and trends
3. Risk Assessment Dashboard - Display risk assessment and alerts
4. Wallet Tracking Dashboard - Manage blacklists and track wallet activities

Usage:
    streamlit run dashboard.py

Note: Requires streamlit to be installed (`pip install streamlit plotly`)
"""

import asyncio
import json
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional

# Import the bridge to MCP servers
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_integration import SolanaAgentKitBridge

# Configure page
st.set_page_config(
    page_title="Solana Token Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for storing data between reruns
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

# Initialize the bridge
@st.cache_resource
def get_bridge():
    return SolanaAgentKitBridge()

bridge = get_bridge()

# Helper function to run async functions
def run_async(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

# Sidebar navigation
st.sidebar.title("Solana Token Analysis")
st.sidebar.image("https://solana.com/src/img/branding/solanaLogoMark.svg", width=100)

# Navigation
tabs = ["Token Analysis", "Pattern Detection", "Risk Assessment", "Wallet Tracking"]
st.session_state.active_tab = st.sidebar.radio("Navigation", tabs, index=tabs.index(st.session_state.active_tab))

# Input for token address (common across tabs)
token_address = st.sidebar.text_input(
    "Token Address", 
    value="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # Default to USDC
    help="Enter the Solana token address (mint) to analyze"
)

st.sidebar.divider()

# Search trending tokens
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

# Display trending tokens if available
if st.session_state.trending_tokens:
    st.sidebar.subheader("Trending Tokens")
    for token in st.session_state.trending_tokens[:5]:  # Show top 5
        if st.sidebar.button(f"{token['symbol']} - {token['name']}", key=f"trending_{token['address']}"):
            token_address = token['address']
            # Trigger refresh of data
            st.experimental_rerun()

# Main content area based on selected tab
if st.session_state.active_tab == "Token Analysis":
    st.title("Token Analysis Dashboard")
    
    # Fetch token data
    if token_address:
        if token_address not in st.session_state.token_data:
            with st.status("Fetching token data..."):
                try:
                    token_info = run_async(bridge.get_token_info(token_address))
                    price_data = run_async(bridge.get_token_price(token_address))
                    
                    # Store in session state
                    st.session_state.token_data[token_address] = {
                        "info": token_info,
                        "price": price_data,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                except Exception as e:
                    st.error(f"Error fetching token data: {str(e)}")
        
        # Display token information
        if token_address in st.session_state.token_data:
            token_data = st.session_state.token_data[token_address]
            token_info = token_data["info"]
            
            # Token overview
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
                
                # Create a pie chart for token distribution if available
                if 'distribution' in holders and holders['distribution']:
                    dist_data = holders['distribution']
                    labels = [f"Wallet {i+1}" for i in range(len(dist_data))]
                    values = [d['percentage'] for d in dist_data]
                    
                    # Add "Others" category if the sum is less than 100%
                    if sum(values) < 100:
                        labels.append("Others")
                        values.append(100 - sum(values))
                    
                    fig = px.pie(values=values, names=labels, title="Token Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Refresh button
            if st.button("Refresh Token Data"):
                # Clear the stored data for this token to force refresh
                if token_address in st.session_state.token_data:
                    del st.session_state.token_data[token_address]
                st.experimental_rerun()

            # Price chart (simulated for demo)
            st.subheader("Price History")
            st.write("*Note: This is simulated price data for demonstration purposes*")
            
            # Generate simulated price data for the last 30 days
            days = 30
            dates = [datetime.datetime.now() - datetime.timedelta(days=i) for i in range(days)]
            dates.reverse()
            
            # Create a sinusoidal pattern with some random noise
            base_price = float(market.get('price_usd', 1.0))
            amplitude = base_price * 0.2  # 20% variation
            prices = [
                base_price + amplitude * np.sin(i / 5) + amplitude * 0.5 * np.random.randn()
                for i in range(days)
            ]
            
            # Ensure no negative prices
            prices = [max(0.0001, p) for p in prices]
            
            # Create the price chart
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
            
            # Volume chart (simulated)
            st.subheader("Trading Volume")
            
            # Generate simulated volume data
            base_volume = float(market.get('volume_24h', 10000))
            volumes = [
                base_volume * (1 + 0.5 * np.random.randn())
                for _ in range(days)
            ]
            
            # Ensure no negative volumes
            volumes = [max(1, v) for v in volumes]
            
            # Create the volume chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=dates,
                y=volumes,
                name='Volume',
                marker=dict(color='#14F195')
            ))
            fig.update_layout(
                title=f"{token_info.get('symbol', 'Token')} Trading Volume (30 Days)",
                xaxis_title="Date",
                yaxis_title="Volume (USD)",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

elif st.session_state.active_tab == "Pattern Detection":
    st.title("Pattern Detection Dashboard")
    
    # Setup columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Pattern Analysis Options")
        
        # Time period selector
        time_period = st.selectbox(
            "Time Period",
            options=["24h", "7d", "30d"],
            index=1,  # Default to 7d
            help="Select time period for pattern analysis"
        )
        
        # Pattern type selector for market patterns
        pattern_type = st.selectbox(
            "Pattern Type",
            options=["all", "pump_dump", "accumulation", "distribution", "breakout"],
            index=0,  # Default to all
            help="Select specific pattern type to look for"
        )
        
        # Confidence threshold
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Set minimum confidence threshold for pattern detection"
        )
        
        # Analyze button for token patterns
        if st.button("Analyze Token Patterns"):
            with st.status("Analyzing token patterns..."):
                try:
                    pattern_data = run_async(bridge.analyze_token_pattern(token_address, time_period))
                    st.session_state.pattern_data[token_address] = pattern_data
                    st.success(f"Pattern analysis complete!")
                except Exception as e:
                    st.error(f"Error analyzing token patterns: {str(e)}")
        
        # Button for market-wide patterns
        if st.button("Detect Market Patterns"):
            with st.status("Detecting market patterns..."):
                try:
                    market_data = run_async(bridge.detect_market_patterns(pattern_type, min_confidence))
                    st.session_state.market_patterns = market_data.get("detected_patterns", [])
                    st.success(f"Detected {len(st.session_state.market_patterns)} market patterns")
                except Exception as e:
                    st.error(f"Error detecting market patterns: {str(e)}")
        
        # Price prediction
        st.subheader("Price Prediction")
        time_horizon = st.selectbox(
            "Time Horizon",
            options=["1h", "24h", "7d"],
            index=1,  # Default to 24h
            help="Select time horizon for price prediction"
        )
        
        if st.button("Get Price Prediction"):
            with st.status("Generating price prediction..."):
                try:
                    prediction = run_async(bridge.get_price_prediction(token_address, time_horizon))
                    
                    # Display the prediction
                    st.write(f"**Current Price:** {prediction.get('current_price', 'Unknown')}")
                    st.write(f"**Predicted Price:** {prediction.get('predicted_price', 'Unknown')}")
                    st.write(f"**Expected Change:** {prediction.get('price_change', 'Unknown')}")
                    st.write(f"**Direction:** {prediction.get('direction', 'Unknown')}")
                    st.write(f"**Confidence:** {prediction.get('confidence', 0):.2f}")
                    st.write(f"**Prediction made at:** {prediction.get('prediction_made_at', '')}")
                    
                    # Warning about predictions
                    st.info(prediction.get('disclaimer', 'This prediction is for educational purposes only. Do not make investment decisions based solely on this information.'))
                except Exception as e:
                    st.error(f"Error generating price prediction: {str(e)}")
    
    with col2:
        # Display token patterns if available
        if token_address in st.session_state.pattern_data:
            pattern_data = st.session_state.pattern_data[token_address]
            st.subheader(f"Detected Patterns for {pattern_data.get('token_address', token_address)}")
            
            # Summary
            st.write(f"**Analysis Summary:** {pattern_data.get('analysis_summary', 'No summary available')}")
            
            # Display patterns
            patterns = pattern_data.get("patterns", [])
            if patterns:
                for i, pattern in enumerate(patterns):
                    with st.expander(f"Pattern {i+1}: {pattern.get('type', 'Unknown')} (Confidence: {pattern.get('confidence', 0):.2f})", expanded=i==0):
                        st.write(f"**ID:** {pattern.get('id', 'Unknown')}")
                        st.write(f"**Type:** {pattern.get('type', 'Unknown')}")
                        st.write(f"**Description:** {pattern.get('description', 'No description available')}")
                        st.write(f"**Detected at:** {pattern.get('detected_at', 'Unknown')}")
                        
                        # If there are indicators, show them
                        if 'indicators' in pattern:
                            indicators = pattern['indicators']
                            st.write("**Indicators:**")
                            for key, value in indicators.items():
                                st.write(f"- {key.replace('_', ' ').title()}: {value}")
                        
                        # Add feedback option
                        st.divider()
                        st.write("**Provide Feedback:**")
                        feedback_accuracy = st.slider(
                            "Accuracy Rating", 
                            min_value=0.0, 
                            max_value=1.0, 
                            value=0.5, 
                            step=0.1,
                            key=f"accuracy_{pattern.get('id', i)}"
                        )
                        feedback_comments = st.text_area(
                            "Comments",
                            key=f"comments_{pattern.get('id', i)}"
                        )
                        if st.button("Submit Feedback", key=f"feedback_{pattern.get('id', i)}"):
                            try:
                                feedback_result = run_async(bridge.provide_pattern_feedback(
                                    pattern.get('id', ''), token_address, feedback_accuracy, feedback_comments
                                ))
                                st.success("Feedback submitted successfully!")
                            except Exception as e:
                                st.error(f"Error submitting feedback: {str(e)}")
            else:
                st.info("No patterns detected for this token in the selected time period.")
            
            # Pattern visualization (simulated for demo)
            st.subheader("Pattern Visualization")
            st.write("*Note: This is a simulated visualization for demonstration purposes*")
            
            # Generate dates for the chart (last 30 days)
            days = 30
            dates = [datetime.datetime.now() - datetime.timedelta(days=i) for i in range(days)]
            dates.reverse()
            
            # Create price data with the pattern
            pattern_type = pattern_data.get("patterns", [{}])[0].get("type", "accumulation") if patterns else "accumulation"
            
            # Generate different price patterns based on the detected pattern type
            if pattern_type == "accumulation":
                # Accumulation pattern - gradually increasing prices
                prices = [10 + i * 0.2 + np.random.randn() for i in range(days)]
            elif pattern_type == "distribution":
                # Distribution pattern - gradually decreasing prices
                prices = [20 - i * 0.15 + np.random.randn() for i in range(days)]
            elif pattern_type == "breakout":
                # Breakout pattern - sudden increase after consolidation
                prices = [15 + np.random.randn() for i in range(days-5)]
                prices.extend([15 + (i+1) * 1.5 + np.random.randn() for i in range(5)])
            elif pattern_type == "pump_dump":
                # Pump and dump pattern
                prices = [15 + np.random.randn() for i in range(days-10)]
                prices.extend([15 + i * 2 + np.random.randn() for i in range(5)])
                prices.extend([25 - i * 2 + np.random.randn() for i in range(5)])
            else:
                # Default pattern
                prices = [15 + 5 * np.sin(i / 5) + np.random.randn() for i in range(days)]
            
            # Create the price chart with pattern visualization
            fig = go.Figure()
            
            # Add the price line
            fig.add_trace(go.Scatter(
                x=dates, 
                y=prices,
                mode='lines',
                name='Price',
                line=dict(color='#9945FF', width=2)
            ))
            
            # Add annotations and shapes based on pattern type
            if pattern_type == "accumulation":
                # Add annotation for accumulation zone
                fig.add_annotation(
                    x=dates[int(days/2)],
                    y=prices[int(days/2)],
                    text="Accumulation Zone",
                    showarrow=True,
                    arrowhead=1
                )
                # Add horizontal rectangle for accumulation zone
                fig.add_shape(
                    type="rect",
                    x0=dates[5],
                    y0=min(prices[5:25]) - 0.5,
                    x1=dates[25],
                    y1=max(prices[5:25]) + 0.5,
                    line=dict(color="green", width=1, dash="dash"),
                    fillcolor="rgba(0, 255, 0, 0.1)"
                )
            elif pattern_type == "distribution":
                # Add annotation for distribution zone
                fig.add_annotation(
                    x=dates[int(days/2)],
                    y=prices[int(days/2)],
                    text="Distribution Zone",
                    showarrow=True,
                    arrowhead=1
                )
                # Add horizontal rectangle for distribution zone
                fig.add_shape(
                    type="rect",
                    x0=dates[5],
                    y0=min(prices[5:25]) - 0.5,
                    x1=dates[25],
                    y1=max(prices[5:25]) + 0.5,
                    line=dict(color="red", width=1, dash="dash"),
                    fillcolor="rgba(255, 0, 0, 0.1)"
                )
            elif pattern_type == "breakout":
                # Add annotation for breakout
                fig.add_annotation(
                    x=dates[days-3],
                    y=prices[days-3],
                    text="Breakout",
                    showarrow=True,
                    arrowhead=1
                )
                # Add vertical line at breakout point
                fig.add_shape(
                    type="line",
                    x0=dates[days-5],
                    y0=min(prices),
                    x1=dates[days-5],
                    y1=max(prices),
                    line=dict(color="blue", width=2, dash="dash")
                )
            elif pattern_type == "pump_dump":
                # Add annotations for pump and dump
                fig.add_annotation(
                    x=dates[days-5],
                    y=prices[days-5],
                    text="Pump",
                    showarrow=True,
                    arrowhead=1
                )
                fig.add_annotation(
                    x=dates[days-2],
                    y=prices[days-2],
                    text="Dump",
                    showarrow=True,
                    arrowhead=1
                )
            
            fig.update_layout(
                title=f"Pattern Visualization: {pattern_type.replace('_', ' ').title()}",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display market patterns if available
        if st.session_state.market_patterns:
            st.subheader("Market-Wide Patterns")
            
            for i, pattern in enumerate(st.session_state.market_patterns):
                with st.expander(f"Market Pattern: {pattern.get('type', 'Unknown')} (Confidence: {pattern.get('confidence', 0):.2f})"):
                    st.write(f"**Type:** {pattern.get('type', 'Unknown')}")
                    st.write(f"**Description:** {pattern.get('description', 'No description available')}")
                    st.write(f"**Detected at:** {pattern.get('detected_at', 'Unknown')}")
                    
                    # Display affected tokens if available
                    if 'affected_tokens' in pattern and pattern['affected_tokens']:
                        st.write("**Affected Tokens:**")
                        affected_df = pd.DataFrame(pattern['affected_tokens'])
                        st.dataframe(affected_df)
                    
                    # Display suggested action if available
                    if 'suggested_action' in pattern:
                        st.info(f"**Suggested Action:** {pattern['suggested_action']}")

elif st.session_state.active_tab == "Risk Assessment":
    st.title("Risk Assessment Dashboard")
    
    # Setup columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Risk Analysis Options")
        
        # Option to force refresh assessment
        force_refresh = st.checkbox("Force Refresh Assessment", value=False)
        
        # Button to assess token risk
        if st.button("Assess Token Risk"):
            with st.status("Assessing token risk..."):
                try:
                    risk_data = run_async(bridge.assess_token_risk(token_address, force_refresh))
                    st.session_state.risk_data[token_address] = risk_data
                    st.success("Risk assessment complete!")
                except Exception as e:
                    st.error(f"Error assessing token risk: {str(e)}")
        
        # Button to get rug pull probability
        if st.button("Calculate Rug Pull Probability"):
            with st.status("Calculating rug pull probability..."):
                try:
                    rug_pull_data = run_async(bridge.get_rug_pull_probability(token_address))
                    if token_address not in st.session_state.risk_data:
                        st.session_state.risk_data[token_address] = {}
                    st.session_state.risk_data[token_address]["rug_pull"] = rug_pull_data
                    st.success("Rug pull probability calculated!")
                except Exception as e:
                    st.error(f"Error calculating rug pull probability: {str(e)}")
        
        # Button to get safety recommendations
        interaction_type = st.selectbox(
            "Interaction Type",
            options=["any", "buy", "sell", "stake", "provide_liquidity"],
            index=0,  # Default to any
            help="Select the type of interaction you plan to have with this token"
        )
        
        if st.button("Get Safety Recommendations"):
            with st.status("Getting safety recommendations..."):
                try:
                    safety_data = run_async(bridge.get_safety_recommendations(token_address, interaction_type))
                    if token_address not in st.session_state.risk_data:
                        st.session_state.risk_data[token_address] = {}
                    st.session_state.risk_data[token_address]["safety"] = safety_data
                    st.success("Safety recommendations received!")
                except Exception as e:
                    st.error(f"Error getting safety recommendations: {str(e)}")
        
        # Button to identify high-risk wallets
        if st.button("Identify High-Risk Wallets"):
            with st.status("Identifying high-risk wallets..."):
                try:
                    wallet_data = run_async(bridge.identify_high_risk_wallets(token_address))
                    if token_address not in st.session_state.risk_data:
                        st.session_state.risk_data[token_address] = {}
                    st.session_state.risk_data[token_address]["high_risk_wallets"] = wallet_data
                    st.success("High-risk wallets identified!")
                except Exception as e:
                    st.error(f"Error identifying high-risk wallets: {str(e)}")
    
    with col2:
        # Display risk assessment if available
        if token_address in st.session_state.risk_data:
            risk_data = st.session_state.risk_data[token_address]
            
            # Core risk assessment
            if "risk_level" in risk_data:
                st.subheader("Risk Assessment Summary")
                
                # Determine color based on risk level
                risk_level = risk_data.get("risk_level", "Unknown")
                risk_color = {
                    "Low": "green",
                    "Medium": "orange",
                    "High": "red"
                }.get(risk_level, "gray")
                
                # Display risk level with colored box
                st.markdown(
                    f"<div style='background-color: {risk_color}; padding: 10px; border-radius: 5px;'>"
                    f"<h3 style='color: white; margin: 0;'>Risk Level: {risk_level}</h3>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                # Display overall risk score
                st.write(f"**Overall Risk Score:** {risk_data.get('overall_risk_score', 0):.2f}")
                
                # Display risk scores as a radar chart
                if "risk_scores" in risk_data:
                    risk_scores = risk_data["risk_scores"]
                    categories = list(risk_scores.keys())
                    values = list(risk_scores.values())
                    
                    # Create the radar chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name='Risk Scores'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=False,
                        title="Risk Score Breakdown"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display concerns
                if "concerns" in risk_data and risk_data["concerns"]:
                    st.subheader("Risk Concerns")
                    for concern in risk_data["concerns"]:
                        st.warning(concern)
            
            # Rug pull probability
            if "rug_pull" in risk_data:
                rug_pull = risk_data["rug_pull"]
                st.subheader("Rug Pull Assessment")
                
                # Display probability gauge chart
                probability = rug_pull.get("probability", 0)
                
                # Create the gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    title = {'text': "Rug Pull Probability"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "red"},
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': probability * 100
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display key factors
                if "key_factors" in rug_pull and rug_pull["key_factors"]:
                    st.subheader("Key Risk Factors")
                    for factor in rug_pull["key_factors"]:
                        st.warning(factor)
            
            # Safety recommendations
            if "safety" in risk_data:
                safety = risk_data["safety"]
                st.subheader("Safety Recommendations")
                
                if "recommendations" in safety:
                    for recommendation in safety["recommendations"]:
                        st.info(recommendation)
            
            # High-risk wallets
            if "high_risk_wallets" in risk_data:
                high_risk_wallets = risk_data["high_risk_wallets"]
                st.subheader("High-Risk Wallets")
                
                if "wallets" in high_risk_wallets and high_risk_wallets["wallets"]:
                    wallet_data = high_risk_wallets["wallets"]
                    wallet_df = pd.DataFrame(wallet_data)
                    st.dataframe(wallet_df)
                else:
                    st.info("No high-risk wallets identified for this token.")

elif st.session_state.active_tab == "Wallet Tracking":
    st.title("Wallet Tracking Dashboard")
    
    # Setup columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Blacklist Management")
        
        # Input for address to check or add
        address_to_check = st.text_input(
            "Address to Check/Add",
            help="Enter a wallet or token address to check or add to the blacklist"
        )
        
        # Radio to select wallet or token
        address_type = st.radio(
            "Address Type",
            options=["wallet", "token"],
            index=0,  # Default to wallet
            help="Select whether this is a wallet or token address"
        )
        
        # Button to check blacklist
        if st.button("Check Blacklist"):
            if address_to_check:
                with st.status("Checking blacklist..."):
                    try:
                        result = run_async(bridge.check_blacklist(address_to_check, address_type))
                        
                        if result.get("is_blacklisted", False):
                            st.error(f"⚠️ **BLACKLISTED** ⚠️ - This {address_type} is on the blacklist!")
                            st.write(f"**Reason:** {result.get('details', {}).get('reason', 'Not specified')}")
                            st.write(f"**Added on:** {result.get('details', {}).get('added_at', 'Unknown')}")
                        else:
                            st.success(f"✅ This {address_type} is not on the blacklist.")
                    except Exception as e:
                        st.error(f"Error checking blacklist: {str(e)}")
            else:
                st.warning("Please enter an address to check.")
        
        # Form to add to blacklist
        st.subheader("Add to Blacklist")
        reason = st.text_area(
            "Reason for Blacklisting",
            help="Provide a reason why this address should be blacklisted"
        )
        
        if st.button("Add to Blacklist"):
            if address_to_check and reason:
                with st.status(f"Adding {address_type} to blacklist..."):
                    try:
                        result = run_async(bridge.add_to_blacklist(address_to_check, address_type, reason))
                        if result.get("success", False):
                            st.success(f"Successfully added {address_to_check} to the {address_type} blacklist.")
                            
                            # Update session state
                            if address_type == "wallet":
                                st.session_state.blacklist["wallets"].append({
                                    "address": address_to_check,
                                    "reason": reason,
                                    "added_at": datetime.datetime.now().isoformat()
                                })
                            else:
                                st.session_state.blacklist["tokens"].append({
                                    "address": address_to_check,
                                    "reason": reason,
                                    "added_at": datetime.datetime.now().isoformat()
                                })
                        else:
                            st.error("Failed to add to blacklist.")
                    except Exception as e:
                        st.error(f"Error adding to blacklist: {str(e)}")
            else:
                st.warning("Please enter both an address and a reason.")
        
        # Option to retrieve wallet tokens
        st.subheader("Wallet Token Analysis")
        wallet_address = st.text_input(
            "Wallet Address",
            help="Enter a wallet address to view its token holdings"
        )
        
        if st.button("Get Wallet Tokens"):
            if wallet_address:
                with st.status("Fetching wallet tokens..."):
                    try:
                        wallet_tokens = run_async(bridge.get_wallet_tokens(wallet_address))
                        st.session_state.wallet_tokens = wallet_tokens
                        st.success(f"Found {len(wallet_tokens.get('tokens', []))} tokens for this wallet.")
                    except Exception as e:
                        st.error(f"Error fetching wallet tokens: {str(e)}")
            else:
                st.warning("Please enter a wallet address.")
    
    with col2:
        # Display blacklist if available
        st.subheader("Current Blacklist")
        
        # Create tabs for wallet and token blacklists
        blacklist_tabs = st.tabs(["Blacklisted Wallets", "Blacklisted Tokens"])
        
        with blacklist_tabs[0]:
            # Display blacklisted wallets
            if st.session_state.blacklist["wallets"]:
                wallet_df = pd.DataFrame(st.session_state.blacklist["wallets"])
                st.dataframe(wallet_df)
            else:
                st.info("No wallets currently blacklisted.")
        
        with blacklist_tabs[1]:
            # Display blacklisted tokens
            if st.session_state.blacklist["tokens"]:
                token_df = pd.DataFrame(st.session_state.blacklist["tokens"])
                st.dataframe(token_df)
            else:
                st.info("No tokens currently blacklisted.")
        
        # Display wallet tokens if available
        if hasattr(st.session_state, 'wallet_tokens') and st.session_state.wallet_tokens:
            st.subheader(f"Tokens Held by {st.session_state.wallet_tokens.get('walletAddress', 'Wallet')}")
            
            tokens = st.session_state.wallet_tokens.get('tokens', [])
            if tokens:
                # Convert to dataframe for easy display
                token_df = pd.DataFrame(tokens)
                st.dataframe(token_df)
                
                # Create a pie chart of token values
                fig = px.pie(
                    token_df,
                    values="amount",
                    names="mint",
                    title="Token Distribution by Amount"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk assessment button for wallet's tokens
                if st.button("Assess Wallet Token Risks"):
                    with st.status("Assessing token risks..."):
                        risk_results = {}
                        for token in tokens[:5]:  # Limit to first 5 tokens to avoid overloading
                            try:
                                token_address = token['mint']
                                risk_data = run_async(bridge.assess_token_risk(token_address))
                                risk_results[token_address] = risk_data
                            except Exception as e:
                                st.error(f"Error assessing token {token['mint']}: {str(e)}")
                        
                        if risk_results:
                            st.success("Risk assessment complete!")
                            
                            # Display risk summary
                            st.subheader("Token Risk Summary")
                            for token_address, risk_data in risk_results.items():
                                risk_level = risk_data.get("risk_level", "Unknown")
                                risk_color = {
                                    "Low": "green",
                                    "Medium": "orange",
                                    "High": "red"
                                }.get(risk_level, "gray")
                                
                                st.markdown(
                                    f"<div style='background-color: {risk_color}; padding: 5px; border-radius: 5px; margin-bottom: 5px;'>"
                                    f"<p style='color: white; margin: 0;'><strong>{token_address[:8]}...</strong> - Risk Level: {risk_level}</p>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
            else:
                st.info("No tokens found for this wallet.")

# Footer
st.markdown("---")
st.caption("Solana Token Analysis Agent Swarm Dashboard | © 2025")
