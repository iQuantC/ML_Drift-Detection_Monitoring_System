import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects data drift using statistical methods:
    - Kolmogorov-Smirnov test for continuous features
    - Population Stability Index (PSI) for categorical features
    """
    
    def __init__(self, reference_data: np.ndarray, feature_names: List[str]):
        """
        Initialize drift detector with reference (training) data
        
        Args:
            reference_data: Reference dataset (training data)
            feature_names: Names of features
        """
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.n_features = reference_data.shape[1]
        
        # Store reference statistics
        self.reference_stats = self._compute_statistics(reference_data)
        
    def _compute_statistics(self, data: np.ndarray) -> Dict:
        """Compute statistics for reference data"""
        stats_dict = {}
        for i in range(data.shape[1]):
            feature_data = data[:, i]
            stats_dict[i] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'quantiles': np.percentile(feature_data, [25, 50, 75])
            }
        return stats_dict
    
    def detect_drift_ks(self, current_data: np.ndarray) -> Dict[str, float]:
        """
        Detect drift using Kolmogorov-Smirnov test
        
        Args:
            current_data: Current production data
            
        Returns:
            Dictionary with drift scores per feature (0-1, higher = more drift)
        """
        drift_scores = {}
        
        for i in range(min(self.n_features, current_data.shape[1])):
            ref_feature = self.reference_data[:, i]
            curr_feature = current_data[:, i]
            
            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(ref_feature, curr_feature)
            
            # Use KS statistic as drift score (0-1)
            drift_scores[self.feature_names[i]] = ks_statistic
            
            logger.info(
                f"Feature '{self.feature_names[i]}': "
                f"KS={ks_statistic:.4f}, p-value={p_value:.4f}"
            )
        
        return drift_scores
    
    def calculate_psi(self, reference: np.ndarray, current: np.ndarray, 
                     bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        Args:
            reference: Reference data
            current: Current data
            bins: Number of bins for discretization
            
        Returns:
            PSI score (0 = no drift, >0.2 = significant drift)
        """
        # Create bins based on reference data
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        
        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=breakpoints)
        curr_hist, _ = np.histogram(current, bins=breakpoints)
        
        # Normalize to get proportions
        ref_prop = ref_hist / len(reference)
        curr_prop = curr_hist / len(current)
        
        # Avoid division by zero
        ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
        curr_prop = np.where(curr_prop == 0, 0.0001, curr_prop)
        
        # Calculate PSI
        psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
        
        return psi
    
    def detect_drift_psi(self, current_data: np.ndarray) -> Dict[str, float]:
        """
        Detect drift using PSI
        
        Args:
            current_data: Current production data
            
        Returns:
            Dictionary with PSI scores per feature
        """
        psi_scores = {}
        
        for i in range(min(self.n_features, current_data.shape[1])):
            ref_feature = self.reference_data[:, i]
            curr_feature = current_data[:, i]
            
            psi = self.calculate_psi(ref_feature, curr_feature)
            psi_scores[self.feature_names[i]] = psi
            
            logger.info(f"Feature '{self.feature_names[i]}': PSI={psi:.4f}")
        
        return psi_scores
    
    def get_drift_status(self, drift_score: float) -> Tuple[str, str]:
        """
        Determine drift status based on score
        
        Args:
            drift_score: Drift score value
            
        Returns:
            Tuple of (status, severity)
        """
        if drift_score < 0.1:
            return "No drift", "info"
        elif drift_score < 0.25:
            return "Minor drift", "warning"
        elif drift_score < 0.5:
            return "Moderate drift", "warning"
        else:
            return "Severe drift", "critical"
