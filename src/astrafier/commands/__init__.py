"""Command line helpers for Astrafier."""

from .train import add_parser as add_train_parser
from .predict import add_parser as add_predict_parser

__all__ = ["add_train_parser", "add_predict_parser"]

