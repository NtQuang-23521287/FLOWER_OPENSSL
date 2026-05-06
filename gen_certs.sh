#!/bin/bash
# Script tạo certificates cho mTLS (mutual TLS)
# Tạo: CA, Server cert, Client cert

set -e

echo "=== Tạo thư mục certs ==="
mkdir -p certs
cd certs

# ──────────────────────────────────────────
# 1. Tạo CA (Certificate Authority)
# ──────────────────────────────────────────
echo ""
echo "=== [1/6] Tạo CA private key ==="
openssl genrsa -out ca.key 4096

echo "=== [2/6] Tạo CA self-signed certificate ==="
openssl req -new -x509 -days 3650 -key ca.key -out ca.crt \
  -subj "/C=VN/ST=HCM/L=HoChiMinh/O=MyOrg/OU=CA/CN=MyRootCA"

# ──────────────────────────────────────────
# 2. Tạo Server certificate
# ──────────────────────────────────────────
echo ""
echo "=== [3/6] Tạo Server private key & CSR ==="
openssl genrsa -out server.key 2048
openssl req -new -key server.key -out server.csr \
  -subj "/C=VN/ST=HCM/L=HoChiMinh/O=MyOrg/OU=Server/CN=localhost"

echo "=== [4/6] Ký Server certificate bằng CA ==="
openssl x509 -req -days 365 \
  -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out server.crt \
  -extfile <(printf "subjectAltName=IP:127.0.0.1,DNS:localhost")

# ──────────────────────────────────────────
# 3. Tạo Client certificate
# ──────────────────────────────────────────
echo ""
echo "=== [5/6] Tạo Client private key & CSR ==="
openssl genrsa -out client.key 2048
openssl req -new -key client.key -out client.csr \
  -subj "/C=VN/ST=HCM/L=HoChiMinh/O=MyOrg/OU=Client/CN=MyClient"

echo "=== [6/6] Ký Client certificate bằng CA ==="
openssl x509 -req -days 365 \
  -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out client.crt

echo ""
echo "✅ Tạo certificates thành công!"
echo "   certs/ca.crt     - CA certificate"
echo "   certs/server.crt - Server certificate"
echo "   certs/server.key - Server private key"
echo "   certs/client.crt - Client certificate"
echo "   certs/client.key - Client private key"