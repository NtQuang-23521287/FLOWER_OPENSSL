"""
server.py — Flower Server với mTLS (mutual TLS qua OpenSSL)

Chạy: python server.py
"""
import sys
import flwr as fl
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters

from config import (
    SERVER_ADDR, NUM_ROUNDS, NUM_CLIENTS,
    MIN_FIT_CLIENTS, MIN_EVAL_CLIENTS,
    CA_CERT, SERVER_CERT, SERVER_KEY,
)
from model import build_model, get_parameters


def get_strategy(initial_parameters):
    """
    FedAvg — thuật toán tổng hợp mặc định của Flower.
    Server nhận weights từ các client, tính trung bình có trọng số
    theo số lượng mẫu, rồi gửi global model mới về cho client.
    """
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,               # Dùng 100% client có sẵn
        fraction_evaluate=1.0,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVAL_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,

        # Callback in kết quả mỗi round
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )


def weighted_average(metrics):
    """
    Tổng hợp metrics từ các client theo trọng số số lượng mẫu.
    metrics: list of (num_examples, {metric_name: value})
    """
    total = sum(n for n, _ in metrics)
    aggregated = {}
    for n, m in metrics:
        for k, v in m.items():
            aggregated[k] = aggregated.get(k, 0) + v * n / total
    return aggregated


def main():
    print("=" * 60)
    print("  Flower mTLS Server")
    print(f"  Địa chỉ  : {SERVER_ADDR}")
    print(f"  Số rounds: {NUM_ROUNDS}")
    print(f"  Clients  : {NUM_CLIENTS}")
    print("=" * 60)

    # Khởi tạo global model — gửi weights ban đầu cho client
    model = build_model()
    initial_params = ndarrays_to_parameters(get_parameters(model))

    strategy  = get_strategy(initial_params)
    server_cfg = ServerConfig(num_rounds=NUM_ROUNDS)

    # ── mTLS certificates ──────────────────────────────────────────────────
    # Flower đọc certificate dưới dạng bytes
    with open(CA_CERT,     "rb") as f: ca_cert_bytes     = f.read()
    with open(SERVER_CERT, "rb") as f: server_cert_bytes = f.read()
    with open(SERVER_KEY,  "rb") as f: server_key_bytes  = f.read()

    print(f"\n[Server] Certificates đã load:")
    print(f"  CA   : {CA_CERT}")
    print(f"  Cert : {SERVER_CERT}")
    print(f"  Key  : {SERVER_KEY}")
    print(f"\n[Server] Đang chờ {NUM_CLIENTS} client kết nối...\n")

    # ── Khởi động server ───────────────────────────────────────────────────
    history = fl.server.start_server(
        server_address=SERVER_ADDR,
        config=server_cfg,
        strategy=strategy,
        certificates=(
            ca_cert_bytes,      # CA certificate — xác thực client
            server_cert_bytes,  # Server certificate
            server_key_bytes,   # Server private key
        ),
    )

    # ── In kết quả sau khi hoàn tất ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Federated Learning hoàn tất!")
    print("=" * 60)

    if history.metrics_distributed:
        print("\nKết quả evaluate theo round:")
        for metric_name in ["accuracy", "loss"]:
            rounds_data = history.metrics_distributed.get(metric_name, [])
            for rnd, val in rounds_data:
                print(f"  Round {rnd:2d} | {metric_name}: {val:.4f}")


if __name__ == "__main__":
    main()