#!/bin/bash
# Setup Bluetooth audio for Lazarillo (Orange Pi 5 / RK3588)
set -e

echo "=== Configurando Bluetooth audio ==="

# Enable HFP in WirePlumber
mkdir -p ~/.config/wireplumber/bluetooth.lua.d/
cat > ~/.config/wireplumber/bluetooth.lua.d/50-bluez-config.lua << 'EOF'
bluez_monitor.properties = {
  ["bluez5.enable-sbc-xq"] = true,
  ["bluez5.enable-msbc"] = true,
  ["bluez5.enable-hw-volume"] = true,
  ["bluez5.headset-roles"] = "[ hsp_hs hsp_ag hfp_hf hfp_ag ]",
}
EOF

# Start services
echo 'orangepi' | sudo -S systemctl enable --now bluetooth ofono 2>/dev/null || true
systemctl --user restart pipewire pipewire-pulse wireplumber 2>/dev/null || true

echo "=== Para conectar auriculares: ==="
echo "  bluetoothctl"
echo "  > power on; agent on; scan on"
echo "  > pair <MAC>; trust <MAC>; connect <MAC>"
echo "=== Listo ==="
