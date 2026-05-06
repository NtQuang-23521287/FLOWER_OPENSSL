"""
client_notls.py — Flower Client KHÔNG có TLS
Dùng để demo Wireshark bắt được payload plaintext.

Chạy:
  python client_notls.py --client-id 1
  python client_notls.py --client-id 2
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import NDArrays, Scalar
from typing import Dict, Tuple

from config import (
    SERVER_HOST, SERVER_PORT, LOCAL_EPOCHS, LEARNING_RATE,
)
from model import build_model, get_parameters, set_parameters
from dataset import load_client_data


class CovidFlowerClient(fl.client.NumPyClient):

    def __init__(self, client_id: int, train_loader: DataLoader,
                 val_loader: DataLoader, device: torch.device):
        self.client_id    = client_id
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.model        = build_model().to(device)

    def get_parameters(self, config: Dict) -> NDArrays:
        return get_parameters(self.model)

    def fit(self, parameters: NDArrays,
            config: Dict) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        set_parameters(self.model, parameters)
        train_loss, train_acc = self._train(LOCAL_EPOCHS)
        print(f"[Client {self.client_id}] fit done | "
              f"loss={train_loss:.4f} acc={train_acc:.4f}")
        return (
            get_parameters(self.model),
            len(self.train_loader.dataset),
            {"loss": train_loss, "accuracy": train_acc},
        )

    def evaluate(self, parameters: NDArrays,
                 config: Dict) -> Tuple[float, int, Dict[str, Scalar]]:
        set_parameters(self.model, parameters)
        val_loss, val_acc = self._evaluate()
        print(f"[Client {self.client_id}] evaluate | "
              f"loss={val_loss:.4f} acc={val_acc:.4f}")
        return (
            float(val_loss),
            len(self.val_loader.dataset),
            {"accuracy": val_acc, "loss": val_loss},
        )

    def _train(self, epochs: int) -> Tuple[float, float]:
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=LEARNING_RATE,
        )
        total_loss, correct, total = 0.0, 0, 0

        for epoch in range(epochs):
            ep_loss, ep_correct, ep_total = 0.0, 0, 0
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                ep_loss    += loss.item() * images.size(0)
                preds       = outputs.argmax(dim=1)
                ep_correct += (preds == labels).sum().item()
                ep_total   += images.size(0)

            print(f"  [Client {self.client_id}] Epoch {epoch+1}/{epochs} "
                  f"loss={ep_loss/ep_total:.4f} acc={ep_correct/ep_total:.4f}")
            total_loss += ep_loss
            correct    += ep_correct
            total      += ep_total

        return total_loss / total, correct / total

    def _evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss    = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                preds       = outputs.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += images.size(0)

        return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Flower Client (No TLS)")
    parser.add_argument("--client-id", type=int, required=True, choices=[1, 2])
    return parser.parse_args()


def main():
    args      = parse_args()
    client_id = args.client_id
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SERVER_ADDR = f"{SERVER_HOST}:{SERVER_PORT}"

    print("=" * 60)
    print(f"  Flower Client {client_id} — KHÔNG CÓ TLS (plaintext)")
    print(f"  Server : {SERVER_ADDR}")
    print(f"  Device : {device}")
    print("  ⚠️  Dữ liệu truyền không được mã hóa!")
    print("=" * 60)

    print(f"\n[Client {client_id}] Đang load dataset...")
    train_loader, val_loader = load_client_data(client_id)

    print(f"\n[Client {client_id}] Kết nối đến server (NO TLS)...\n")

    flower_client = CovidFlowerClient(client_id, train_loader, val_loader, device)

    fl.client.start_client(
        server_address=SERVER_ADDR,
        client=flower_client.to_client(),
        # root_certificates=None  ← không truyền → plaintext gRPC
    )

    print(f"\n[Client {client_id}] Hoàn tất. Kết nối đóng.")


if __name__ == "__main__":
    main()