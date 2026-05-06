"""
server_notls.py — Flower Server KHÔNG có TLS
Dùng để demo Wireshark bắt được payload plaintext.

Chạy: python server_notls.py
"""
import flwr as fl
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters

from config import (
    SERVER_HOST, SERVER_PORT, NUM_ROUNDS,
    NUM_CLIENTS, MIN_FIT_CLIENTS, MIN_EVAL_CLIENTS,
)
from model import build_model, get_parameters


def weighted_average(metrics):
    total = sum(n for n, _ in metrics)
    aggregated = {}
    for n, m in metrics:
        for k, v in m.items():
            aggregated[k] = aggregated.get(k, 0) + v * n / total
    return aggregated


def main():
    SERVER_ADDR = f"{SERVER_HOST}:{SERVER_PORT}"

    print("=" * 60)
    print("  Flower Server — KHÔNG CÓ TLS (plaintext)")
    print(f"  Địa chỉ  : {SERVER_ADDR}")
    print(f"  Số rounds: {NUM_ROUNDS}")
    print("  ⚠️  Mọi dữ liệu truyền đều có thể bị đọc bởi Wireshark!")
    print("=" * 60)

    model = build_model()
    initial_params = ndarrays_to_parameters(get_parameters(model))

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVAL_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_params,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )

    print(f"\n[Server] Đang chờ {NUM_CLIENTS} client kết nối (NO TLS)...\n")

    history = fl.server.start_server(
        server_address=SERVER_ADDR,
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        # certificates=None  ← không truyền → plaintext gRPC
    )

    print("\n" + "=" * 60)
    print("  Federated Learning hoàn tất!")
    print("=" * 60)

    if history.metrics_distributed:
        print("\nKết quả evaluate theo round:")
        for metric_name in ["accuracy", "loss"]:
            for rnd, val in history.metrics_distributed.get(metric_name, []):
                print(f"  Round {rnd:2d} | {metric_name}: {val:.4f}")


if __name__ == "__main__":
    main()