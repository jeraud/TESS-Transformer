"""Command line helpers for Astrafier."""

from .train import add_parser as add_train_parser
from .predict import add_parser as add_predict_parser
from .preprocess import add_parser as add_preprocess_parser
from .split import add_parser as add_split_parser

__all__ = ["add_train_parser", "add_predict_parser", "add_preprocess_parser", "add_split_parser"]

