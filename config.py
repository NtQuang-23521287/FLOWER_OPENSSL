"""
config.py — Cấu hình chung cho toàn bộ hệ thống Flower mTLS
"""
import os

# ── Đường dẫn gốc project ──────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CERT_DIR     = os.path.join(BASE_DIR, "certs")
DATASET_DIR  = os.path.join(BASE_DIR, "COVID-19_Radiography_Dataset")

# ── Certificates (tạo bằng gen_certs.ps1) ─────────────────────────────────
CA_CERT      = os.path.join(CERT_DIR, "ca.crt")
SERVER_CERT  = os.path.join(CERT_DIR, "server.crt")
SERVER_KEY   = os.path.join(CERT_DIR, "server.key")
CLIENT1_CERT = os.path.join(CERT_DIR, "client1.crt")
CLIENT1_KEY  = os.path.join(CERT_DIR, "client1.key")
CLIENT2_CERT = os.path.join(CERT_DIR, "client2.crt")
CLIENT2_KEY  = os.path.join(CERT_DIR, "client2.key")

# ── Server ─────────────────────────────────────────────────────────────────
SERVER_HOST  = "127.0.0.1"
SERVER_PORT  = 8443
SERVER_ADDR  = f"{SERVER_HOST}:{SERVER_PORT}"

# ── Federated Learning ──────────────────────────────────────────────────────
NUM_ROUNDS        = 5       # Số vòng FL
NUM_CLIENTS       = 2       # Tổng số client
MIN_FIT_CLIENTS   = 2       # Tối thiểu client tham gia train mỗi round
MIN_EVAL_CLIENTS  = 2       # Tối thiểu client tham gia evaluate

# ── Dataset ────────────────────────────────────────────────────────────────
# Phân chia non-IID:
#   Client 1 → COVID + Normal
#   Client 2 → Viral Pneumonia + Normal
CLASSES = ["COVID", "Normal", "Viral Pneumonia"]

CLIENT_DATA = {
    1: ["COVID", "Normal"],
    2: ["Viral Pneumonia", "Normal"],
}

# Giới hạn ảnh mỗi class mỗi client (None = dùng hết)
MAX_IMAGES_PER_CLASS = 500

# ── Model / Training ────────────────────────────────────────────────────────
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32
LOCAL_EPOCHS = 3
LEARNING_RATE = 1e-3
NUM_CLASSES  = len(CLASSES)