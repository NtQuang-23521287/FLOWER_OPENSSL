# Script tạo certificates cho mTLS (mutual TLS)
# Tạo: CA, Server cert, Client1 cert, Client2 cert

$ErrorActionPreference = "Stop"

# ──────────────────────────────────────────
# 1. Tạo thư mục certs
# ──────────────────────────────────────────
Write-Host "=== Tạo thư mục certs ==="
New-Item -ItemType Directory -Force -Path "certs"
Set-Location -Path "certs"

# ──────────────────────────────────────────
# 2. Tạo CA (Certificate Authority)
# ──────────────────────────────────────────
Write-Host ""
Write-Host "=== [1/8] Tạo CA private key ==="
& openssl genrsa -out ca.key 4096

Write-Host "=== [2/8] Tạo CA self-signed certificate ==="
& openssl req -new -x509 -days 3650 -key ca.key -out ca.crt `
  -subj "/C=VN/ST=HCM/L=HoChiMinh/O=MyOrg/OU=CA/CN=MyRootCA"

# ──────────────────────────────────────────
# 3. Tạo Server certificate
# ──────────────────────────────────────────
Write-Host ""
Write-Host "=== [3/8] Tạo Server private key & CSR ==="
& openssl genrsa -out server.key 2048
& openssl req -new -key server.key -out server.csr `
  -subj "/C=VN/ST=HCM/L=HoChiMinh/O=MyOrg/OU=Server/CN=localhost"

Write-Host "=== [4/8] Ký Server certificate bằng CA ==="
$extContent = @"
[SAN]
subjectAltName=IP:127.0.0.1,DNS:localhost
"@
$extContent | Out-File -FilePath extfile.cnf -Encoding ascii -NoNewline:$false

& openssl x509 -req -days 365 `
  -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial `
  -out server.crt -extfile extfile.cnf -extensions SAN

# ──────────────────────────────────────────
# 4. Tạo Client1 certificate
# ──────────────────────────────────────────
Write-Host ""
Write-Host "=== [5/8] Tạo Client1 private key & CSR ==="
& openssl genrsa -out client1.key 2048
& openssl req -new -key client1.key -out client1.csr `
  -subj "/C=VN/ST=HCM/L=HoChiMinh/O=MyOrg/OU=Client/CN=MyClient1"

Write-Host "=== [6/8] Ký Client1 certificate bằng CA ==="
& openssl x509 -req -days 365 `
  -in client1.csr -CA ca.crt -CAkey ca.key -CAcreateserial `
  -out client1.crt

# ──────────────────────────────────────────
# 5. Tạo Client2 certificate
# ──────────────────────────────────────────
Write-Host ""
Write-Host "=== [7/8] Tạo Client2 private key & CSR ==="
& openssl genrsa -out client2.key 2048
& openssl req -new -key client2.key -out client2.csr `
  -subj "/C=VN/ST=HCM/L=HoChiMinh/O=MyOrg/OU=Client/CN=MyClient2"

Write-Host "=== [8/8] Ký Client2 certificate bằng CA ==="
& openssl x509 -req -days 365 `
  -in client2.csr -CA ca.crt -CAkey ca.key -CAcreateserial `
  -out client2.crt

# ──────────────────────────────────────────
# 6. Dọn file tạm
# ──────────────────────────────────────────
Remove-Item -Force *.csr, *.srl, extfile.cnf -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "✅ Tạo certificates thành công!"
Write-Host "   certs/ca.crt      - CA certificate"
Write-Host "   certs/server.crt  - Server certificate"
Write-Host "   certs/server.key  - Server private key"
Write-Host "   certs/client1.crt - Client1 certificate"
Write-Host "   certs/client1.key - Client1 private key"
Write-Host "   certs/client2.crt - Client2 certificate"
Write-Host "   certs/client2.key - Client2 private key"