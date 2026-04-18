#!/usr/bin/env python3
"""Test audio input/output on Orange Pi."""
import subprocess, shutil, sys

def check_playback():
    for cmd in ['aplay', 'ffplay', 'mpg123']:
        if shutil.which(cmd):
            print(f"✅ Playback: {cmd} disponible")
            return cmd
    print("❌ Sin player de audio (instala: sudo apt install alsa-utils)")
    return None

def check_capture():
    result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
    if 'card' in result.stdout:
        print("✅ Captura: dispositivos encontrados")
        print(result.stdout[:200])
    else:
        print("❌ Sin dispositivos de captura")

def list_devices():
    print("\n=== Dispositivos de salida ===")
    subprocess.run(['aplay', '-l'], capture_output=False)
    print("\n=== Dispositivos de entrada ===")
    subprocess.run(['arecord', '-l'], capture_output=False)

if __name__ == '__main__':
    check_playback()
    check_capture()
    list_devices()
