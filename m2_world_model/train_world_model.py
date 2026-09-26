"""Train Module 2's GRU World Model from the frozen M1 tensors."""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from world_model import WorldModel


def load_schema(data_dir):
    with open(os.path.join(data_dir, "feature_schema.json")) as f:
        return json.load(f)


def load_split(data_dir, split):
    seq = np.load(os.path.join(data_dir, f"state_sequences_{split}.npy"))
    nxt = np.load(os.path.join(data_dir, f"next_state_{split}.npy"))
    return seq, nxt


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_mse = total_mae = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            bs = xb.size(0)
            total_mse += loss_fn(pred, yb).item() * bs
            total_mae += torch.mean(torch.abs(pred - yb)).item() * bs
            n += bs
    return total_mse / n, total_mae / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=".", help="M1 output directory")
    p.add_argument("--out-dir", default="./output")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    schema = load_schema(args.data_dir)
    input_dim = schema["state_feature_dim"]
    sequence_length = schema["sequence_length"]

    X_train, y_train = load_split(args.data_dir, "train")
    X_val, y_val = load_split(args.data_dir, "val")
    assert X_train.shape[1:] == (sequence_length, input_dim)
    assert y_train.shape[1] == input_dim
    assert X_val.shape[1:] == (sequence_length, input_dim)
    assert y_val.shape[1] == input_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    config = {
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "sequence_length": sequence_length,
        "model_type": "GRU",
    }
    model = WorldModel(**config, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    stale = 0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = n = 0
        started = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        train_mse = total / n
        val_mse, val_mae = evaluate(model, val_loader, device, loss_fn)
        row = {
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "seconds": time.time() - started,
        }
        history.append(row)
        print(row)

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "world_model.pt"))
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(
        torch.load(
            os.path.join(args.out_dir, "world_model.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    val_mse, val_mae = evaluate(model, val_loader, device, loss_fn)

    metrics = {
        "val_mse": val_mse,
        "val_mae": val_mae,
        "best_epoch": best_epoch,
        "training_epochs": len(history),
        "seed": args.seed,
    }

    test_seq = os.path.join(args.data_dir, "state_sequences_test.npy")
    if os.path.exists(test_seq):
        X_test, y_test = load_split(args.data_dir, "test")
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
            batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        test_mse, test_mae = evaluate(model, test_loader, device, loss_fn)
        metrics.update({"test_mse": test_mse, "test_mae": test_mae})

    with open(os.path.join(args.out_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.out_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
