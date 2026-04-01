"""
ClusterPeek — SLURM cluster GPU/node dashboard
Backend: FastAPI + SSH ControlMaster (no stored passwords)
"""

import re
import subprocess
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="ClusterPeek")

CONFIG_PATH = Path(__file__).parent / "clusters.yaml"
STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def get_clusters() -> dict:
    return load_config().get("clusters", {})


# ---------------------------------------------------------------------------
# SSH helpers (ControlMaster-based — no password handling)
# ---------------------------------------------------------------------------

def _ssh_target(cfg: dict) -> str:
    user = cfg.get("user", "").strip()
    host = cfg["host"]
    return f"{user}@{host}" if user else host


def _control_path(cfg: dict) -> str:
    return str(Path(cfg.get("control_path", f"~/.ssh/control-{cfg['host']}")).expanduser())


def is_connected(cfg: dict) -> bool:
    """Return True if a live ControlMaster socket exists for this cluster."""
    result = subprocess.run(
        ["ssh", "-O", "check", "-o", f"ControlPath={_control_path(cfg)}", _ssh_target(cfg)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def run_remote(cfg: dict, command: str, timeout: int = 30) -> tuple[str, str, int]:
    """Run a shell command on the remote cluster via the ControlMaster socket."""
    ssh_cmd = [
        "ssh",
        "-o", f"ControlPath={_control_path(cfg)}",
        "-o", "ControlMaster=no",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        _ssh_target(cfg),
        command,
    ]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout, result.stderr, result.returncode


# ---------------------------------------------------------------------------
# SLURM parsing
# ---------------------------------------------------------------------------

def _parse_gres(gres_str: str) -> tuple[str, int]:
    """Extract (gpu_type, count) from a gres string like 'gpu:a100:4'."""
    if not gres_str or gres_str in ("(null)", "N/A", ""):
        return "none", 0
    m = re.search(r"gpu:([^:(]+)[:(]?:?(\d+)", gres_str)
    if m:
        return m.group(1).lower(), int(m.group(2))
    # bare 'gpu:4' with no type
    m2 = re.search(r"gpu:(\d+)", gres_str)
    if m2:
        return "gpu", int(m2.group(1))
    return "none", 0


def _parse_cpu_state(s: str) -> dict:
    """Parse SLURM CPU state string A/I/O/T into a dict."""
    parts = s.strip().split("/")
    try:
        return {
            "allocated": int(parts[0]),
            "idle":      int(parts[1]),
            "other":     int(parts[2]),
            "total":     int(parts[3]),
        }
    except (IndexError, ValueError):
        return {"allocated": 0, "idle": 0, "other": 0, "total": 0}


def fetch_cluster_status(cfg: dict) -> dict:
    """
    Query the cluster with scontrol show node --oneliner and return structured data.
    Falls back to sinfo if scontrol is unavailable.
    """
    cmd = "scontrol show node --oneliner 2>/dev/null"
    stdout, stderr, rc = run_remote(cfg, cmd)

    if rc != 0 or not stdout.strip():
        return {"error": f"scontrol failed (rc={rc}): {stderr.strip() or 'no output'}"}

    nodes = []
    for line in stdout.strip().splitlines():
        node = _parse_scontrol_node_line(line)
        if node:
            nodes.append(node)

    if not nodes:
        return {"error": "No node data returned from scontrol"}

    return {
        "nodes": nodes,
        "gpu_summary": _build_gpu_summary(nodes),
        "partition_summary": _build_partition_summary(nodes),
    }


def _parse_scontrol_node_line(line: str) -> dict | None:
    """Parse one line of `scontrol show node --oneliner` output."""
    def field(name: str) -> str:
        m = re.search(rf"{name}=(\S+)", line)
        return m.group(1) if m else ""

    node_name = field("NodeName")
    if not node_name:
        return None

    state_raw = field("State")
    # Normalize SLURM state (may have suffixes like +DRAIN, *POWER, etc.)
    state = re.split(r"[+*#~$@!]", state_raw)[0].upper()

    gres_raw = field("Gres")
    gres_used_raw = field("GresUsed")

    gpu_type, gpu_total = _parse_gres(gres_raw)
    _, gpu_used = _parse_gres(gres_used_raw)

    # Partitions is comma-separated
    partitions = [p for p in field("Partitions").split(",") if p]

    cpu_tot_str = field("CPUTot")
    cpu_load_str = field("CPULoad")
    cfg_tres = field("CfgTRES")
    alloc_tres = field("AllocTRES")

    cpu_total = int(cpu_tot_str) if cpu_tot_str.isdigit() else 0
    try:
        cpu_load = float(cpu_load_str)
    except ValueError:
        cpu_load = 0.0

    # AllocTRES gives us allocated CPUs reliably
    alloc_cpu = 0
    m_cpu = re.search(r"cpu=(\d+)", alloc_tres)
    if m_cpu:
        alloc_cpu = int(m_cpu.group(1))

    return {
        "name":        node_name,
        "state":       state,
        "state_raw":   state_raw,
        "partitions":  partitions,
        "gpu_type":    gpu_type,
        "gpu_total":   gpu_total,
        "gpu_used":    gpu_used,
        "gpu_free":    max(0, gpu_total - gpu_used),
        "cpu_total":   cpu_total,
        "cpu_alloc":   alloc_cpu,
        "cpu_idle":    max(0, cpu_total - alloc_cpu),
        "cpu_load":    round(cpu_load, 2),
    }


def _build_gpu_summary(nodes: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for n in nodes:
        if n["gpu_type"] == "none" or n["gpu_total"] == 0:
            continue
        t = n["gpu_type"]
        if t not in summary:
            summary[t] = {"total": 0, "used": 0, "free": 0, "node_count": 0}
        summary[t]["total"] += n["gpu_total"]
        summary[t]["used"]  += n["gpu_used"]
        summary[t]["free"]  += n["gpu_free"]
        summary[t]["node_count"] += 1
    return summary


def _build_partition_summary(nodes: list[dict]) -> dict:
    parts: dict[str, dict] = {}
    for n in nodes:
        for p in n["partitions"]:
            if p not in parts:
                parts[p] = {
                    "node_count": 0,
                    "gpu_total": 0,
                    "gpu_free": 0,
                    "cpu_total": 0,
                    "cpu_idle": 0,
                    "states": {},
                }
            d = parts[p]
            d["node_count"] += 1
            d["gpu_total"]  += n["gpu_total"]
            d["gpu_free"]   += n["gpu_free"]
            d["cpu_total"]  += n["cpu_total"]
            d["cpu_idle"]   += n["cpu_idle"]
            s = n["state"]
            d["states"][s] = d["states"].get(s, 0) + 1
    return parts


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/clusters")
def api_clusters():
    clusters = get_clusters()
    result = {}
    for name, cfg in clusters.items():
        cp = _control_path(cfg)
        connected = is_connected(cfg)
        result[name] = {
            "label":        cfg.get("label", name),
            "host":         cfg["host"],
            "user":         cfg.get("user", ""),
            "notes":        cfg.get("notes", ""),
            "connected":    connected,
            "control_path": cp,
            "connect_cmd":  f"ssh -M -S {cp} -N -f {_ssh_target(cfg)}",
        }
    return result


@app.get("/api/clusters/{cluster_name}/status")
def api_cluster_status(cluster_name: str):
    clusters = get_clusters()
    if cluster_name not in clusters:
        raise HTTPException(status_code=404, detail="Cluster not found")

    cfg = clusters[cluster_name]
    if not is_connected(cfg):
        cp = _control_path(cfg)
        return {
            "connected": False,
            "connect_cmd": f"ssh -M -S {cp} -N -f {_ssh_target(cfg)}",
        }

    data = fetch_cluster_status(cfg)
    data["connected"] = True
    return data


@app.get("/api/clusters/{cluster_name}/disconnect")
def api_disconnect(cluster_name: str):
    clusters = get_clusters()
    if cluster_name not in clusters:
        raise HTTPException(status_code=404, detail="Cluster not found")
    cfg = clusters[cluster_name]
    cp = _control_path(cfg)
    subprocess.run(
        ["ssh", "-O", "exit", "-o", f"ControlPath={cp}", _ssh_target(cfg)],
        capture_output=True,
    )
    return {"ok": True}


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8765, reload=True)
