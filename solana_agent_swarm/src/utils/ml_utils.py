"""
Machine learning utilities for the Solana Token Analysis Agent Swarm.
Provides integration with scikit-learn and PyTorch for pattern recognition and risk assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
from loguru import logger

# Import scikit-learn components
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available. Machine learning features will be limited.")
    SKLEARN_AVAILABLE = False

# Import PyTorch components
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Deep learning features will be disabled.")
    PYTORCH_AVAILABLE = False

class PatternDetector:
    """Pattern detection for token price and volume behavior."""
    
    def __init__(self):
        """Initialize the pattern detector."""
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.anomaly_detector = IsolationForest(contamination=0.05) if SKLEARN_AVAILABLE else None
        self.pattern_classifier = RandomForestClassifier() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        
    def train(self, historical_data: List[Dict[str, Any]], patterns: Optional[List[str]] = None) -> bool:
        """
        Train the pattern detector on historical token data.
        
        Args:
            historical_data: List of historical data points with price, volume, etc.
            patterns: Optional list of pattern labels for supervised learning
            
        Returns:
            Success status of training
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Cannot train pattern detector - scikit-learn not available")
            return False
            
        try:
            # Extract features from historical data
            features = self._extract_features(historical_data)
            
            # Normalize features
            scaled_features = self.scaler.fit_transform(features)
            
            # Train anomaly detector
            self.anomaly_detector.fit(scaled_features)
            
            # Train pattern classifier if labels are available
            if patterns and len(patterns) == len(historical_data):
                self.pattern_classifier.fit(scaled_features, patterns)
            
            self.is_trained = True
            logger.info("Successfully trained pattern detector")
            return True
            
        except Exception as e:
            logger.error(f"Error training pattern detector: {str(e)}")
            return False
    
    def detect_anomalies(self, token_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in token data.
        
        Args:
            token_data: List of token data points to analyze
            
        Returns:
            List of anomaly results with scores and flags
        """
        if not SKLEARN_AVAILABLE or not self.is_trained:
            logger.warning("Cannot detect anomalies - model not available or not trained")
            return []
            
        try:
            # Extract and scale features
            features = self._extract_features(token_data)
            scaled_features = self.scaler.transform(features)
            
            # Get anomaly scores (-1 for anomalies, 1 for normal)
            raw_scores = self.anomaly_detector.decision_function(scaled_features)
            
            # Convert to normalized scores (0-1, higher is more anomalous)
            normalized_scores = 1.0 - (raw_scores + 1) / 2
            
            # Create result objects
            results = []
            for i, score in enumerate(normalized_scores):
                results.append({
                    "timestamp": token_data[i].get("timestamp", datetime.now().isoformat()),
                    "anomaly_score": float(score),
                    "is_anomaly": score > 0.7,  # Threshold for anomaly detection
                    "data_point": token_data[i]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    def classify_pattern(self, token_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Classify token behavior into known patterns.
        
        Args:
            token_data: List of token data points to analyze
            
        Returns:
            Pattern classification result or None if not available
        """
        if not SKLEARN_AVAILABLE or not self.is_trained:
            logger.warning("Cannot classify pattern - model not available or not trained")
            return None
            
        try:
            # Check if pattern classifier is trained
            if not hasattr(self.pattern_classifier, "classes_"):
                logger.warning("Pattern classifier not trained with labeled data")
                return None
                
            # Extract and scale features
            features = self._extract_features(token_data)
            scaled_features = self.scaler.transform(features)
            
            # Predict pattern
            pattern_probs = self.pattern_classifier.predict_proba(scaled_features)
            pattern_idx = np.argmax(pattern_probs, axis=1)[0]
            
            return {
                "pattern_type": self.pattern_classifier.classes_[pattern_idx],
                "confidence": float(pattern_probs[0][pattern_idx]),
                "all_probabilities": {
                    cls: float(prob) 
                    for cls, prob in zip(self.pattern_classifier.classes_, pattern_probs[0])
                }
            }
            
        except Exception as e:
            logger.error(f"Error classifying pattern: {str(e)}")
            return None
    
    def _extract_features(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract numerical features from token data.
        
        Args:
            data: List of token data points
            
        Returns:
            NumPy array of features
        """
        # Example feature extraction - customize based on available data
        features = []
        
        for item in data:
            # Basic price and volume features
            price = item.get("price", 0)
            volume = item.get("volume", 0)
            
            # Get additional metrics if available
            market_cap = item.get("market_cap", 0)
            liquidity = item.get("liquidity", 0)
            
            # Create feature vector
            feature_vector = [
                price, 
                volume,
                market_cap,
                liquidity,
                # Add more features as needed
            ]
            
            features.append(feature_vector)
            
        return np.array(features)


class TokenPricePredictor:
    """Deep learning model for token price prediction using PyTorch."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 50, num_layers: int = 2, output_size: int = 1):
        """
        Initialize the token price predictor.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of LSTM layers
            output_size: Number of output values
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Initialize PyTorch model if available
        if PYTORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = LSTMPriceModel(input_size, hidden_size, num_layers, output_size).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.MSELoss()
            self.is_initialized = True
        else:
            logger.warning("PyTorch not available. TokenPricePredictor will be disabled.")
            self.is_initialized = False
    
    async def train(self, features: np.ndarray, targets: np.ndarray, 
                   epochs: int = 100, batch_size: int = 64) -> bool:
        """
        Train the price prediction model.
        
        Args:
            features: Input features (sequence data)
            targets: Target values
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Success status of training
        """
        if not PYTORCH_AVAILABLE or not self.is_initialized:
            logger.warning("Cannot train model - PyTorch not available")
            return False
            
        try:
            # Prepare data
            X = torch.tensor(features, dtype=torch.float32).to(self.device)
            y = torch.tensor(targets, dtype=torch.float32).to(self.device)
            
            # Create DataLoader
            dataset = TensorDataset(X, y)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for X_batch, y_batch in data_loader:
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    
                    # Backward and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss/len(data_loader):.6f}")
            
            logger.info(f"Training completed after {epochs} epochs")
            return True
            
        except Exception as e:
            logger.error(f"Error training price prediction model: {str(e)}")
            return False
    
    async def predict(self, features: np.ndarray) -> Optional[np.ndarray]:
        """
        Make price predictions using the trained model.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Predicted values
        """
        if not PYTORCH_AVAILABLE or not self.is_initialized:
            logger.warning("Cannot predict - PyTorch not available")
            return None
            
        try:
            # Prepare data
            X = torch.tensor(features, dtype=torch.float32).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X)
                return predictions.cpu().numpy()
                
        except Exception as e:
            logger.error(f"Error making price predictions: {str(e)}")
            return None
    
    async def save_model(self, path: str) -> bool:
        """
        Save the trained model to a file.
        
        Args:
            path: File path to save model
            
        Returns:
            Success status
        """
        if not PYTORCH_AVAILABLE or not self.is_initialized:
            logger.warning("Cannot save model - PyTorch not available")
            return False
            
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_config': {
                    'input_size': self.input_size,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'output_size': self.output_size
                }
            }, path)
            
            logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    async def load_model(self, path: str) -> bool:
        """
        Load a trained model from a file.
        
        Args:
            path: File path to load model from
            
        Returns:
            Success status
        """
        if not PYTORCH_AVAILABLE or not self.is_initialized:
            logger.warning("Cannot load model - PyTorch not available")
            return False
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Check if configuration matches
            config = checkpoint.get('model_config', {})
            if config.get('input_size') != self.input_size or config.get('output_size') != self.output_size:
                logger.warning(f"Model configuration mismatch. Expected input_size={self.input_size}, output_size={self.output_size}. "
                              f"Got input_size={config.get('input_size')}, output_size={config.get('output_size')}")
                # Recreate model with loaded configuration
                self.input_size = config.get('input_size', self.input_size)
                self.hidden_size = config.get('hidden_size', self.hidden_size)
                self.num_layers = config.get('num_layers', self.num_layers)
                self.output_size = config.get('output_size', self.output_size)
                self.model = LSTMPriceModel(self.input_size, self.hidden_size, self.num_layers, self.output_size).to(self.device)
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            # Load model and optimizer state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False


class LSTMPriceModel(nn.Module):
    """LSTM model for token price prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of LSTM layers
            output_size: Number of output values
        """
        super(LSTMPriceModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = self.dropout(out[:, -1, :])
        
        # Pass through fully connected layer
        out = self.fc(out)
        return out
