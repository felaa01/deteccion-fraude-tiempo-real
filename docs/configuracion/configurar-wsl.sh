#!/usr/bin/env bash
# Ejecutar dentro de Ubuntu (WSL), NO en PowerShell:  bash configurar-wsl.sh
set -euo pipefail

echo "==> Verificando systemd (necesario para Docker Engine)"
if [ "$(ps -p 1 -o comm=)" != "systemd" ]; then
  echo "systemd no está activado. Agregá esto a /etc/wsl.conf, ejecutá 'wsl --shutdown' y volvé a intentar:"
  echo "  [boot]"
  echo "  systemd=true"
  exit 1
fi

echo "==> Paquetes base"
sudo apt-get update
sudo apt-get install -y git make curl build-essential ca-certificates

echo "==> Docker Engine (dentro de WSL, sin Docker Desktop, para ahorrar memoria)"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo usermod -aG docker "$USER"

echo "==> Limitar el tamaño de los logs de Docker para que no llenen el disco"
sudo mkdir -p /etc/docker
echo '{ "log-driver": "json-file", "log-opts": { "max-size": "10m", "max-file": "3" } }' \
  | sudo tee /etc/docker/daemon.json >/dev/null
sudo systemctl restart docker

echo "==> uv (gestor de Python y dependencias)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "==> Git"
git config --global core.autocrlf input
git config --global init.defaultBranch main

mkdir -p "$HOME/proyectos"

echo
echo "Listo. Cerrá esta terminal y abrí una nueva para que se aplique el grupo docker."
echo "Después comprobá con:  docker run --rm hello-world"
echo "Trabajá dentro de ~/proyectos (sistema de archivos de Linux), nunca en /mnt/c."
