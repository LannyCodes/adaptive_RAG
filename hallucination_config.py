"""
Hallucination Detector Configuration
Configure which detection method to use
"""

# Detection method: 'vectara', 'nli', 'lightweight', or 'hybrid' (recommended)
# 注意: lightweight 是新添加的轻量级方案，无需特殊权限
HALLUCINATION_DETECTION_METHOD = "lightweight"

# Thresholds
VECTARA_HALLUCINATION_THRESHOLD = 0.5  # Score above this = hallucination
NLI_CONTRADICTION_THRESHOLD = 0.3  # Percentage of contradictions to flag

# Performance settings
USE_GPU = True  # Use GPU if available
BATCH_SIZE = 8  # For batch processing

# Fallback behavior
FALLBACK_TO_LLM = True  # If professional detectors fail, use LLM method
