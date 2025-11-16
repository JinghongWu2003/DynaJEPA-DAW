"""Model registry for DynaJEPA-STL10."""

from .encoder_backbone import ConvEncoder
from .jepa import JEPAModel
from .autoencoder import ConvAutoencoder

__all__ = ["ConvEncoder", "JEPAModel", "ConvAutoencoder"]

