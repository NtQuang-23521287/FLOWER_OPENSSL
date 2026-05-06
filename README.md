# Federated Learning với Flower + mTLS + COVID-19 X-Ray

Hệ thống phân loại ảnh X-quang COVID-19 theo kiến trúc **Federated Learning**,
bảo mật truyền thông bằng **mutual TLS (mTLS)** dùng OpenSSL.

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────┐
│                   Flower Server                     │
│  • FedAvg aggregation                               │
│  • mTLS: server.crt / server.key / ca.crt           │
│  • Gửi global model → nhận updated weights          │
└──────────────┬──────────────────┬───────────────────┘
               │  mTLS (gRPC)     │  mTLS (gRPC)
    ┌──────────▼──────┐    ┌──────▼───────────┐
    │   Client 1      │    │   Client 2        │
    │  client1.crt    │    │  client2.crt      │
    │  COVID + Normal │    │  VPneumonia+Normal│
    │  MobileNetV2    │    │  MobileNetV2      │
    └─────────────────┘    └───────────────────┘
```

### Phân chia data (non-IID)

| Client   | Classes                       | Mô phỏng              |
|----------|-------------------------------|-----------------------|
| Client 1 | COVID + Normal                | Bệnh viện COVID chuyên|
| Client 2 | Viral Pneumonia + Normal      | Bệnh viện đa khoa     |

---

## Cấu trúc thư mục

```
flower_mtls/
├── config.py          # Cấu hình chung (paths, hyperparams)
├── model.py           # MobileNetV2 + helper functions
├── dataset.py         # Dataset loader cho từng client
├── server.py          # Flower Server mTLS
├── client.py          # Flower Client mTLS (dùng --client-id)
├── certs/             # Đặt certificates vào đây
│   ├── ca.crt
│   ├── server.crt / server.key
│   ├── client1.crt / client1.key
│   └── client2.crt / client2.key
└── COVID-19_Radiography_Dataset/   # Dataset Kaggle
    ├── COVID/images/
    ├── Normal/images/
    └── Viral Pneumonia/images/
```

---

## Yêu cầu cài đặt

```bash
python -m venv flower
flower\Scripts\activate
pip install -r requirements.txt
```

> Python 3.9+, PyTorch 2.x, Flower 1.x

---

## Hướng dẫn chạy

### Bước 1 — Tạo certificates (chạy 1 lần)

```powershell
# Trong PowerShell, tại thư mục flower_mtls/
.\generate-certs.ps1
```

### Bước 2 — Chạy Server (Terminal 1)

```bash
python server.py
```

Output mẫu:
```
============================================================
  Flower mTLS Server
  Địa chỉ  : 127.0.0.1:8443
  Số rounds: 5
  Clients  : 2
============================================================
[Server] Đang chờ 2 client kết nối...
```

### Bước 3 — Chạy Client 1 (Terminal 2)

```bash
python client.py --client-id 1
```

### Bước 4 — Chạy Client 2 (Terminal 3)

```bash
python client.py --client-id 2
```

---

## Luồng hoạt động mỗi Round

```
Round N:
  Server ──[global weights]──► Client 1
  Server ──[global weights]──► Client 2
         Client 1 train local (3 epochs)
         Client 2 train local (3 epochs)
  Client 1 ──[updated weights]──► Server
  Client 2 ──[updated weights]──► Server
         Server FedAvg aggregation
         Server evaluate trên tất cả clients
```

## Kiểm tra trên WireShark
Filter để xem traffic:

```bash
tcp.port == 8443
```
Xem payload gRPC/HTTP2:
Click chuột phải vào packet → Decode As → chọn HTTP2 → Wireshark sẽ decode được frame gRPC và hiện payload dạng text.

---

## Bảo mật mTLS

| Bước              | Nội dung                                     |
|-------------------|----------------------------------------------|
| Server auth       | Server gửi `server.crt`, client xác minh qua `ca.crt` |
| Client auth       | Client gửi `client1/2.crt`, server xác minh qua `ca.crt` |
| Encryption        | Toàn bộ weights truyền qua kênh TLS mã hóa  |
| Thư viện          | Python `ssl` module (wrapper OpenSSL)        |

---

## Điều chỉnh config

Mở `config.py` để thay đổi:

| Tham số               | Mặc định | Ý nghĩa                    |
|-----------------------|----------|----------------------------|
| `NUM_ROUNDS`          | 5        | Số vòng Federated Learning |
| `LOCAL_EPOCHS`        | 3        | Epochs train mỗi client    |
| `MAX_IMAGES_PER_CLASS`| 500      | Giới hạn ảnh/class/client  |
| `BATCH_SIZE`          | 32       | Batch size                 |
| `LEARNING_RATE`       | 1e-3     | Learning rate Adam         |

---

## Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|----------|
| `CERTIFICATE_VERIFY_FAILED` | Certificate không khớp CA | Chạy lại `gen_certs.ps1` |
| `FileNotFoundError: certs/...` | Thiếu file cert | Kiểm tra thư mục `certs/` |
| `FileNotFoundError: COVID-19...` | Sai đường dẫn dataset | Sửa `DATASET_DIR` trong `config.py` |
| Client chờ mãi | Server chưa chạy | Chạy `server.py` trước |
| CUDA out of memory | GPU không đủ VRAM | Giảm `BATCH_SIZE` xuống 16 |
