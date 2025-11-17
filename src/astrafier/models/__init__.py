"""Model components for the Astrafier project."""

from .module import AstrafierModule
from .light_curve_classifier import LightCurveEncoder
from .head import ClassificationHead

__all__ = ["AstrafierModule", "LightCurveEncoder", "ClassificationHead"]

