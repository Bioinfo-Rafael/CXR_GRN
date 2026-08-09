#!/usr/bin/env python3
"""Run one real NIH batch through MRGL and backpropagate once."""

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
    parser.add_argument("--lr", type=float, default=0.01, help="Paper learning rate")
    parser.add_argument(
        "--optimizer-step",
        action="store_true",
        help="Also call optimizer.step(); backward is always executed",
    )
    args = parser.parse_args()

    train_loader, model, semantic_adapter, device = prepare(args)
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    batch = move_batch(next(iter(train_loader)), device)
    optimizer.zero_grad(set_to_none=True)
    output, details = forward_one_batch(model, semantic_adapter, batch)
    loss = criterion(output, batch["target"])
    loss.backward()
    if args.optimizer_step:
        optimizer.step()
    print_shapes(batch, output, details, loss)
    print("loss.backward() ok")
    print("optimizer.step()", "executed" if args.optimizer_step else "ready")


if __name__ == "__main__":
    main()
