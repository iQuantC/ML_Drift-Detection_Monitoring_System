import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGenerator:
    """
    Generates synthetic data for ML model simulation
    Simulates both reference (training) and production data with drift
    """
    
    def __init__(self, n_features: int = 5, random_state: int = 42):
        """
        Initialize data generator
        
        Args:
            n_features: Number of features to generate
            random_state: Random seed for reproducibility
        """
        self.n_features = n_features
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Define feature names
        self.feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Original distribution parameters (training data)
        self.original_means = np.random.uniform(-1, 1, n_features)
        self.original_stds = np.random.uniform(0.5, 2.0, n_features)
        
        logger.info(f"Initialized DataGenerator with {n_features} features")
    
    def generate_reference_data(self, n_samples: int = 1000) -> np.ndarray:
        """
        Generate reference (training) data
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            numpy array of shape (n_samples, n_features)
        """
        data = np.random.normal(
            loc=self.original_means,
            scale=self.original_stds,
            size=(n_samples, self.n_features)
        )
        
        logger.info(f"Generated {n_samples} reference samples")
        return data
    
    def generate_production_data(
        self, 
        n_samples: int = 100,
        drift_severity: float = 0.0,
        drift_features: list = None
    ) -> np.ndarray:
        """
        Generate production data with optional drift
        
        Args:
            n_samples: Number of samples to generate
            drift_severity: Drift amount (0 = no drift, 1 = high drift)
            drift_features: List of feature indices to apply drift to (None = all)
            
        Returns:
            numpy array of shape (n_samples, n_features)
        """
        if drift_features is None:
            drift_features = list(range(self.n_features))
        
        # Start with data from original distribution
        means = self.original_means.copy()
        stds = self.original_stds.copy()
        
        # Apply drift to specified features
        for idx in drift_features:
            if idx < self.n_features:
                # Shift mean
                means[idx] += drift_severity * self.original_stds[idx] * 2
                # Increase variance
                stds[idx] *= (1 + drift_severity * 0.5)
        
        data = np.random.normal(
            loc=means,
            scale=stds,
            size=(n_samples, self.n_features)
        )
        
        return data
    
    def simulate_prediction(self, features: np.ndarray) -> float:
        """
        Simulate ML model prediction (simple linear combination)
        
        Args:
            features: Feature vector
            
        Returns:
            Predicted value
        """
        # Simple linear model for simulation
        weights = np.array([0.5, -0.3, 0.8, 0.2, -0.6][:self.n_features])
        prediction = np.dot(features, weights) + np.random.normal(0, 0.1)
        
        # Apply sigmoid to get probability
        probability = 1 / (1 + np.exp(-prediction))
        
        return probability
    
    def get_feature_names(self) -> list:
        """Return feature names"""
        return self.feature_names
