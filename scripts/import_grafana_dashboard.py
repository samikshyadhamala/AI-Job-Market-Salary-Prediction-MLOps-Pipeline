#!/usr/bin/env python3
"""Import Grafana dashboard into running Grafana instance."""

import json
import os
import sys
import requests

GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
GRAFANA_USER = os.getenv("GRAFANA_USER", "admin")
GRAFANA_PASS = os.getenv("GRAFANA_PASS", "")
DASHBOARD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "grafana", "provisioning", "dashboards", "drift_monitoring_dashboard.json",
)


def import_dashboard(path: str, user: str, password: str):
    if not password:
        raise RuntimeError("GRAFANA_PASS env var is required")

    with open(path) as f:
        dashboard = json.load(f)

    dashboard.pop("apiVersion", None)
    payload = {"dashboard": dashboard, "overwrite": True, "folderId": 0}
    resp = requests.post(
        f"{GRAFANA_URL}/api/dashboards/db",
        json=payload,
        auth=(user, password),
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if resp.status_code in (200, 201):
        print(f"Imported: {dashboard.get('title', path)}")
        return True
    print(f"Failed {path}: {resp.status_code} {resp.text}")
    return False


def main():
    user = GRAFANA_USER
    password = GRAFANA_PASS
    if not password:
        print("Set GRAFANA_PASS env var with your Grafana password")
        sys.exit(1)
    if not os.path.isfile(DASHBOARD_PATH):
        print(f"Dashboard file not found: {DASHBOARD_PATH}")
        sys.exit(1)

    ok = import_dashboard(DASHBOARD_PATH, user, password)
    print(f"Imported {'1' if ok else '0'} dashboard(s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
