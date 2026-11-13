#!/usr/bin/env bash
# Creates a double-clickable "ECG MI Demo" icon on the Ubuntu desktop.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESK="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
F="$DESK/ECG-MI-Demo.desktop"
cat > "$F" <<EOT
[Desktop Entry]
Type=Application
Name=ECG MI Demo
Comment=Myocardial infarction screening from 12-lead ECG
Exec=bash "$APP_DIR/start_demo.sh"
Icon=utilities-system-monitor
Terminal=true
EOT
chmod +x "$F" "$APP_DIR/start_demo.sh"
gio set "$F" metadata::trusted true 2>/dev/null
echo "Created $F"
echo "If Ubuntu shows it as untrusted: right-click the icon -> 'Allow Launching'."
