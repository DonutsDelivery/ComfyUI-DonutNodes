#!/usr/bin/env python3

"""
Test script to debug compatibility analysis issues
"""

import sys
import os

# Add the current directory to the path  
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add ComfyUI directory to path
comfy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, comfy_dir)

# Try to import scipy first to check dependencies
try:
    import scipy
    print(f"SciPy available: {scipy.__version__}")
except ImportError:
    print("SciPy not available - some functions may be limited")

try:
    import numpy as np
    print(f"NumPy available: {np.__version__}")
except ImportError as e:
    print(f"Failed to import numpy: {e}")
    sys.exit(1)

# Create a simple mock torch if not available
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"Failed to import torch: {e}")
    print("Creating mock torch for testing...")
    
    class MockTensor:
        def __init__(self, data):
            if isinstance(data, list):
                self.data = np.array(data, dtype=np.float32)
            else:
                self.data = data.astype(np.float32) if hasattr(data, 'astype') else np.array(data, dtype=np.float32)
            self.shape = self.data.shape
            
        def flatten(self):
            return MockTensor(self.data.flatten())
            
        def detach(self):
            return self
            
        def cpu(self):
            return self
            
        def float(self):
            return self
            
        def item(self):
            return float(self.data.item()) if self.data.size == 1 else float(self.data.flat[0])
            
        def numel(self):
            return self.data.size
            
        def abs(self):
            return MockTensor(np.abs(self.data))
            
        def mean(self):
            return MockTensor(np.mean(self.data))
            
        def std(self):
            return MockTensor(np.std(self.data))
            
        def min(self):
            return MockTensor(np.min(self.data))
            
        def max(self):
            return MockTensor(np.max(self.data))
            
        def __add__(self, other):
            if isinstance(other, MockTensor):
                return MockTensor(self.data + other.data)
            return MockTensor(self.data + other)
            
        def __sub__(self, other):
            if isinstance(other, MockTensor):
                return MockTensor(self.data - other.data)
            return MockTensor(self.data - other)
            
        def __mul__(self, other):
            if isinstance(other, MockTensor):
                return MockTensor(self.data * other.data)
            return MockTensor(self.data * other)
            
    class MockTorch:
        @staticmethod
        def randn(*shape):
            return MockTensor(np.random.randn(*shape))
            
        @staticmethod
        def zeros(*shape):
            return MockTensor(np.zeros(shape))
            
        @staticmethod
        def tensor(data, **kwargs):
            return MockTensor(data)
            
        @staticmethod
        def norm(tensor):
            if isinstance(tensor, MockTensor):
                return MockTensor(np.linalg.norm(tensor.data))
            return MockTensor(np.linalg.norm(tensor))
            
        @staticmethod
        def cosine_similarity(a, b, dim=0):
            if isinstance(a, MockTensor):
                a_data = a.data
            else:
                a_data = a
            if isinstance(b, MockTensor):
                b_data = b.data
            else:
                b_data = b
                
            # Compute cosine similarity
            dot_product = np.sum(a_data * b_data)
            norm_a = np.linalg.norm(a_data)
            norm_b = np.linalg.norm(b_data)
            
            if norm_a == 0 or norm_b == 0:
                return MockTensor(0.0)
            
            cos_sim = dot_product / (norm_a * norm_b)
            return MockTensor(cos_sim)
            
        @staticmethod
        def isnan(tensor):
            if isinstance(tensor, MockTensor):
                return np.isnan(tensor.data).any()
            return np.isnan(tensor)
            
    torch = MockTorch()

from block_compatibility_detector import BlockCompatibilityDetector

def test_compatibility_analysis():
    """Test the compatibility analysis with simple tensors"""
    print("Testing BlockCompatibilityDetector...")
    
    # Create detector
    detector = BlockCompatibilityDetector(
        statistical_threshold=3.0,
        cosine_similarity_threshold=0.1,
        magnitude_ratio_threshold=50.0,
        outlier_zscore_threshold=4.0
    )
    
    # Test with simple tensors
    result = detector.test_compatibility_analysis()
    
    print(f"\nTest completed. Got {len(result['compatibility_scores'])} compatibility scores")
    
    if result['compatibility_scores']:
        print(f"Score range: [{min(result['compatibility_scores']):.3f}, {max(result['compatibility_scores']):.3f}]")
        print(f"All scores: {result['compatibility_scores']}")
    else:
        print("ERROR: No compatibility scores were generated!")
    
    return result

if __name__ == "__main__":
    test_compatibility_analysis()