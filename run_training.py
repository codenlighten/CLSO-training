#!/usr/bin/env python3
"""
CLSO Training Runner

Wrapper script to run CLSO training with proper path setup.
This handles all import issues automatically.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
repo_root = Path(__file__).parent
src_path = repo_root / 'src'
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(src_path))

# Now import and run the training script
if __name__ == "__main__":
    # Import after path is set
    from src.train_clso import main
    
    # Run the main training function
    main()
