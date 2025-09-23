"""Command line interface for the spam detection pipeline.

Usage examples:
  python -m main eda --data data/emails.csv
  python -m main train --data data/emails.csv
  python -m main evaluate
"""
from __future__ import annotations
import argparse

import sys

import data_processing as dp
import eda
import train as train_module
import evaluate as eval_module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spam detection pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_preprocess = sub.add_parser("preprocess", help="Preprocess data and save artifacts")
    p_preprocess.add_argument("--data", default="data/emails.csv")

    p_eda = sub.add_parser("eda", help="Run exploratory data analysis")
    p_eda.add_argument("--data", default="data/emails.csv")

    p_train = sub.add_parser("train", help="Train models")
    p_train.add_argument("--data", default="data/emails.csv")

    p_eval = sub.add_parser("evaluate", help="Evaluate models on test set")
    p_eval.add_argument("--prefix", default="spam")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "preprocess":
        dp.preprocess_data(args.data)
    elif args.command == "eda":
        eda.run_eda(args.data)
    elif args.command == "train":
        train_module.train(args.data)
    elif args.command == "evaluate":
        eval_module.evaluate(prefix=args.prefix)
    else:
        parser.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
