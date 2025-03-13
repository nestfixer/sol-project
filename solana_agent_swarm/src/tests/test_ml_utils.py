"""
Tests for machine learning utilities.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.ml_utils import PatternDetector, TokenPricePredictor


@pytest.fixture
def sample_token_data():
    """Create sample token data for testing."""
    base_time = datetime.now()
    base_price = 10.0
    
    # Create 30 days of simulated price data
    data = []
    for i in range(30):
        # Create some price movement patterns
        if i < 10:
            # Uptrend
            change = 0.05 + (np.random.random() * 0.05)
        elif i < 20:
            # Downtrend
            change = -0.03 - (np.random.random() * 0.05)
        else:
            # Sideways with slight uptrend
            change = 0.01 + (np.random.random() * 0.04) - 0.02
            
        # Calculate new price
        base_price = base_price * (1 + change)
        
        # Add some volume
        volume = 100000 + (np.random.random() * 50000)
        
        # Add market cap and liquidity
        market_cap = base_price * 1000000
        liquidity = market_cap * 0.1
        
        # Create data point
        data.append({
            "timestamp": (base_time + timedelta(days=i)).isoformat(),
            "price": base_price,
            "volume": volume,
            "market_cap": market_cap,
            "liquidity": liquidity,
            "direction": "up" if change > 0 else "down"
        })
        
    return data


@pytest.mark.asyncio
async def test_pattern_detector(sample_token_data):
    """Test pattern detector functionality."""
    try:
        # Import scikit-learn to check if it's available
        import sklearn
        has_sklearn = True
    except ImportError:
        has_sklearn = False
        
    if not has_sklearn:
        pytest.skip("scikit-learn not available - skipping test")
        
    # Create pattern detector
    detector = PatternDetector()
    
    # Train detector
    pattern_labels = ["uptrend", "downtrend", "sideways"] * 10
    success = detector.train(sample_token_data, pattern_labels)
    
    # Check training
    assert success, "Training should succeed if scikit-learn is available"
    assert detector.is_trained, "Detector should be marked as trained after successful training"
    
    # Test anomaly detection
    anomalies = detector.detect_anomalies(sample_token_data[:5])
    assert len(anomalies) == 5, "Should return results for all data points"
    assert "anomaly_score" in anomalies[0], "Results should include anomaly score"
    
    # Test pattern classification
    pattern = detector.classify_pattern(sample_token_data[:1])
    if pattern:  # May be None if classifier not properly trained with labels
        assert "pattern_type" in pattern, "Pattern result should include pattern type"
        assert "confidence" in pattern, "Pattern result should include confidence score"


@pytest.mark.asyncio
async def test_token_price_predictor():
    """Test token price predictor functionality."""
    try:
        # Import PyTorch to check if it's available
        import torch
        has_pytorch = True
    except ImportError:
        has_pytorch = False
        
    if not has_pytorch:
        pytest.skip("PyTorch not available - skipping test")
        
    # Create price predictor
    predictor = TokenPricePredictor(input_size=5, hidden_size=10, num_layers=1, output_size=1)
    
    # Test model initialization
    assert predictor.is_initialized if has_pytorch else not predictor.is_initialized
    
    if has_pytorch:
        # Create simple training data (sequence length = 5, features = 5, samples = 10)
        features = np.random.random((10, 5, 5)).astype(np.float32)  # 10 samples, 5 time steps, 5 features
        targets = np.random.random((10, 1)).astype(np.float32)      # 10 target values
        
        # Test training
        success = await predictor.train(features, targets, epochs=2, batch_size=2)
        assert success, "Training should succeed if PyTorch is available"
        
        # Test prediction
        predictions = await predictor.predict(features)
        assert predictions is not None, "Predictions should be returned if model is trained"
        assert predictions.shape == (10, 1), "Predictions should match expected shape"
