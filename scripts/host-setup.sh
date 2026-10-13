#!/bin/bash
# One-time host setup for serving on both RTX 3090s. Run with sudo, then reboot.
#
#   sudo ./scripts/host-setup.sh && sudo reboot
#
# What it does:
#   1. Installs NVIDIA kernel modules matching the newest installed kernel
#      (the driver was installed for an older kernel, so nvidia-smi fails).
#   2. Installs the docker compose plugin.
#   3. Installs nvidia-container-toolkit and registers it with Docker.
#   4. Disables vfio-card1.service, which reserves the first 3090 for VM
#      passthrough. Re-enable it to give that card back to a VM.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "Run with sudo." >&2
    exit 1
fi

apt-get update
apt-get install -y linux-modules-nvidia-595-open-generic-hwe-26.04 docker-compose-v2 curl gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker

systemctl disable vfio-card1.service

echo ""
echo "Done. Reboot to load the NVIDIA driver on both cards and restart Docker:"
echo "  sudo reboot"
echo "Then check with: nvidia-smi   (should list two RTX 3090s)"
