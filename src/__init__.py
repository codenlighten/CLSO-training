"""
CLSO - Crystalline Latent Space Optimization

A novel neuroevolutionary training framework for energy-efficient LLMs.
"""

from .basis_library import BasisLibrary
from .crystalline_model import CrystallineGPT2, CrystallineLinear
from .genetic_optimizer import GeneticOptimizer, FitnessPredictor

__version__ = "0.1.0"
__author__ = "Gregory J Ward"
__affiliations__ = "SmartLedger.Technology, Codenlighten.org"

__all__ = [
    'BasisLibrary',
    'CrystallineGPT2',
    'CrystallineLinear',
    'GeneticOptimizer',
    'FitnessPredictor',
]
