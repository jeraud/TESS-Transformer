"""Top-level command line entry point for Astrafier."""

from __future__ import annotations

import argparse
from typing import Iterable

from .commands import add_predict_parser, add_preprocess_parser, add_split_parser, add_train_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astrafier", description="Astrafier command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_train_parser(subparsers)
    add_predict_parser(subparsers)
    add_preprocess_parser(subparsers)
    add_split_parser(subparsers)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return
    func(args)


if __name__ == "__main__":
    main()

