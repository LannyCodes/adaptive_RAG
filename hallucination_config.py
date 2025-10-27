"""
Hallucination Detector Configuration
Configure which detection method to use
"""

# Detection method: 'vectara', 'nli', or 'hybrid' (recommended)
HALLUCINATION_DETECTION_METHOD = "hybrid"

# Thresholds
VECTARA_HALLUCINATION_THRESHOLD = 0.5  # Score above this = hallucination
NLI_CONTRADICTION_THRESHOLD = 0.3  # Percentage of contradictions to flag

# Performance settings
USE_GPU = True  # Use GPU if available
BATCH_SIZE = 8  # For batch processing

# Fallback behavior
FALLBACK_TO_LLM = True  # If professional detectors fail, use LLM method
