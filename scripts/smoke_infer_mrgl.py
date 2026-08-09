#!/usr/bin/env python3
"""Run one real NIH batch through MRGL under torch.no_grad()."""

import argparse

import torch

from _mrgl_smoke import (
    add_common_arguments,
    forward_one_batch,
    move_batch,
    prepare,
    print_shapes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()

    test_loader, model, semantic_adapter, device = prepare(args)
    model.eval()
    batch = move_batch(next(iter(test_loader)), device)
    with torch.no_grad():
        output, details = forward_one_batch(model, semantic_adapter, batch)
    print_shapes(batch, output, details)
    print("torch.no_grad() forward ok")


if __name__ == "__main__":
    main()
