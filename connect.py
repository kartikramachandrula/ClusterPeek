#!/usr/bin/env python3
"""Establish SSH ControlMaster sessions for all clusters in clusters.yaml."""
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    config_path = Path(__file__).parent / "clusters.yaml"
    config = yaml.safe_load(config_path.read_text()) or {}
    clusters = config.get("clusters", {})

    if not clusters:
        print("No clusters found in clusters.yaml")
        sys.exit(1)

    for name, cfg in clusters.items():
        user = cfg.get("user", "").strip()
        host = cfg["host"]
        target = f"{user}@{host}" if user else host
        control = str(Path(cfg.get("control_path", f"~/.ssh/control-{host}")).expanduser())

        check = subprocess.run(
            ["ssh", "-O", "check", "-o", f"ControlPath={control}", target],
            capture_output=True,
        )
        if check.returncode == 0:
            print(f"[{name}] Already connected to {target}")
            continue

        print(f"[{name}] Connecting to {target}…")
        print(f"        (You may be prompted for Kerberos / Duo authentication)")
        result = subprocess.run(["ssh", "-M", "-S", control, "-N", "-f", target])
        if result.returncode == 0:
            print(f"[{name}] Connected.")
        else:
            print(f"[{name}] Failed (exit {result.returncode})")


if __name__ == "__main__":
    main()
