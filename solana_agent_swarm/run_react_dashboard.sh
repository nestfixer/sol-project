#!/bin/bash
echo "Installing required dependencies..."
pip install flask flask-cors
echo ""
echo "Starting Solana Token Analysis React Dashboard..."
python src/frontend/dashboard.py --react
