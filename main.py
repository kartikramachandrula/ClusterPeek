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
    result = subprocess.run(
        ["ssh", "-O", "check", "-o", f"ControlPath={_control_path(cfg)}", _ssh_target(cfg)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def run_remote(cfg: dict, command: str, timeout: int = 60) -> tuple[str, str, int]:
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
# SLURM parsing helpers
# ---------------------------------------------------------------------------

def _parse_gres(gres_str: str) -> tuple[str, int]:
    """Extract (gpu_type, count) from gres strings. Handles all SLURM formats:
      gpu:a100:4          gpu:a100:4(S:0)
      gpu:a100:2(IDX:0,1) gpu:2(IDX:0,1)   gpu:4   gpu:0(IDX:N/A)
    """
    if not gres_str or gres_str in ("(null)", "N/A", ""):
        return "none", 0
    # gpu:TYPE:COUNT — type starts with a letter, distinguishing from gpu:COUNT
    m = re.search(r"gpu:([a-zA-Z][^:()]*):(\d+)", gres_str)
    if m:
        return m.group(1).lower(), int(m.group(2))
    # gpu:COUNT (no type label, e.g. GresUsed=gpu:2(IDX:0,1))
    m = re.search(r"gpu:(\d+)", gres_str)
    if m:
        return "gpu", int(m.group(1))
    return "none", 0


def _parse_cpu_state(s: str) -> dict:
    """Parse SLURM cpusstate string A/I/O/T."""
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


def _normalize_state(state_raw: str) -> str:
    return re.split(r"[+*#~$@!]", state_raw)[0].upper() if state_raw else "UNKNOWN"


# ---------------------------------------------------------------------------
# Cluster status via sinfo -N --Format (fixed-width column slicing)
# ---------------------------------------------------------------------------
#
# WHY: scontrol --oneliner formats GresUsed inconsistently across SLURM
# versions (e.g. omitting the GPU type label). sinfo --Format with explicit
# column widths gives us a byte-exact layout we can slice without ambiguity.

_SINFO_WIDTHS = [30, 30, 60, 60, 20, 20]  # node, partition, gres, gresused, state, cpus
_SINFO_FIELDS = ["nodelist", "partition", "gres", "gresused", "statelong", "cpusstate"]
_SINFO_FORMAT = ",".join(f"{f}:{w}" for f, w in zip(_SINFO_FIELDS, _SINFO_WIDTHS))

# Pre-compute cumulative offsets
_SINFO_OFFSETS = [0]
for _w in _SINFO_WIDTHS:
    _SINFO_OFFSETS.append(_SINFO_OFFSETS[-1] + _w)
_SINFO_LINE_LEN = _SINFO_OFFSETS[-1]


def _parse_sinfo_line(line: str) -> dict | None:
    line = line.ljust(_SINFO_LINE_LEN)
    slices = [line[_SINFO_OFFSETS[i]:_SINFO_OFFSETS[i+1]].strip()
              for i in range(len(_SINFO_WIDTHS))]
    node_name, partition, gres, gres_used, state_raw, cpus_str = slices

    if not node_name:
        return None

    gpu_type, gpu_total = _parse_gres(gres)
    _, gpu_used = _parse_gres(gres_used)
    cpu = _parse_cpu_state(cpus_str)

    return {
        "name":      node_name,
        "partition": partition,
        "state":     _normalize_state(state_raw),
        "state_raw": state_raw,
        "gpu_type":  gpu_type,
        "gpu_total": gpu_total,
        "gpu_used":  gpu_used,
        "gpu_free":  max(0, gpu_total - gpu_used),
        "cpu_total": cpu["total"],
        "cpu_alloc": cpu["allocated"],
        "cpu_idle":  cpu["idle"],
    }


def _fetch_node_users(cfg: dict) -> dict[str, list[str]]:
    """Return {node_name: [username, ...]} for all running jobs."""
    # %N = nodelist, %u = username — one row per job
    stdout, _, rc = run_remote(cfg, "squeue -h -t R -o '%N|%u'")
    if rc != 0:
        return {}
    node_users: dict[str, list[str]] = {}
    for line in stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        nodelist, user = parts[0].strip(), parts[1].strip()
        # squeue may give a compressed nodelist (e.g. node[1-3,5])
        # Store against the whole nodelist string; frontend displays it as-is
        for node in _expand_nodelist(nodelist):
            node_users.setdefault(node, [])
            if user not in node_users[node]:
                node_users[node].append(user)
    return node_users


def _expand_nodelist(nodelist: str) -> list[str]:
    """Best-effort expansion of SLURM nodelist into individual names.
    Handles node[1,3-5] style ranges. Falls back to returning the raw string."""
    if "[" not in nodelist:
        return [nodelist] if nodelist else []
    prefix, rest = nodelist.split("[", 1)
    rest = rest.rstrip("]")
    nodes = []
    for part in rest.split(","):
        if "-" in part:
            try:
                lo, hi = part.split("-", 1)
                width = len(lo)
                nodes += [f"{prefix}{str(i).zfill(width)}" for i in range(int(lo), int(hi) + 1)]
            except ValueError:
                nodes.append(f"{prefix}{part}")
        else:
            nodes.append(f"{prefix}{part}")
    return nodes


def fetch_cluster_status(cfg: dict) -> dict:
    cmd = f"sinfo -N -h --Format={_SINFO_FORMAT}"
    stdout, stderr, rc = run_remote(cfg, cmd)

    if rc != 0 or not stdout.strip():
        return {"error": f"sinfo failed (rc={rc}): {stderr.strip() or 'no output'}"}

    rows = [r for line in stdout.splitlines() if (r := _parse_sinfo_line(line))]

    if not rows:
        return {"error": "No node data returned from sinfo"}

    # Deduplicate nodes: keep first row per name; accumulate partitions list
    seen: dict[str, dict] = {}
    for row in rows:
        n = row["name"]
        if n not in seen:
            seen[n] = {**row, "partitions": [row["partition"]]}
            del seen[n]["partition"]
        elif row["partition"] and row["partition"] not in seen[n]["partitions"]:
            seen[n]["partitions"].append(row["partition"])

    # Annotate each node with the users currently running jobs on it
    node_users = _fetch_node_users(cfg)
    for node in seen.values():
        node["users"] = node_users.get(node["name"], [])

    nodes = list(seen.values())

    return {
        "nodes":             nodes,
        "gpu_summary":       _build_gpu_summary(nodes),
        "partition_summary": _build_partition_summary(rows),
    }


def _build_gpu_summary(nodes: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for n in nodes:
        if n["gpu_type"] == "none" or n["gpu_total"] == 0:
            continue
        t = n["gpu_type"]
        if t not in summary:
            summary[t] = {"total": 0, "used": 0, "free": 0, "node_count": 0}
        summary[t]["total"]      += n["gpu_total"]
        summary[t]["used"]       += n["gpu_used"]
        summary[t]["free"]       += n["gpu_free"]
        summary[t]["node_count"] += 1
    return summary


def _build_partition_summary(rows: list[dict]) -> dict:
    """Build partition summary from raw sinfo rows (one per node×partition)."""
    parts: dict[str, dict] = {}
    seen_pairs: set[tuple] = set()

    for row in rows:
        key = (row["partition"], row["name"])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        p = row["partition"]
        if not p:
            continue
        if p not in parts:
            parts[p] = {"node_count": 0, "gpu_total": 0, "gpu_free": 0,
                        "cpu_total": 0, "cpu_idle": 0, "states": {}}
        d = parts[p]
        d["node_count"] += 1
        d["gpu_total"]  += row["gpu_total"]
        d["gpu_free"]   += row["gpu_free"]
        d["cpu_total"]  += row["cpu_total"]
        d["cpu_idle"]   += row["cpu_idle"]
        s = row["state"]
        d["states"][s] = d["states"].get(s, 0) + 1

    return parts


# ---------------------------------------------------------------------------
# My jobs via squeue --me
# ---------------------------------------------------------------------------

def fetch_my_jobs(cfg: dict) -> list[dict]:
    # %i=jobid %j=name %P=partition %T=state %N=nodelist %b=gres/node
    # %C=cpus  %m=min_mem %l=timelimit %S=starttime %r=reason(pending)
    cmd = 'squeue --me -h -o "%i|%j|%P|%T|%N|%b|%C|%m|%l|%S|%r"'
    stdout, stderr, rc = run_remote(cfg, cmd)

    if rc != 0:
        return []

    jobs = []
    for line in stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 11:
            continue
        job_id, name, partition, state, nodelist, gres, cpus, mem, timelimit, start, reason = parts[:11]
        gpu_type, gpus = _parse_gres(gres.strip())
        if gpu_type == "gpu":
            gpu_type = ""  # unlabelled — don't show a type
        jobs.append({
            "job_id":    job_id.strip(),
            "name":      name.strip(),
            "partition": partition.strip(),
            "state":     state.strip(),
            "nodelist":  nodelist.strip(),
            "gres":      gres.strip(),
            "gpu_type":  gpu_type,
            "gpus":      gpus,
            "cpus":      cpus.strip(),
            "mem":       mem.strip(),
            "timelimit": timelimit.strip(),
            "start":     start.strip(),
            "reason":    reason.strip() if state.strip() == "PENDING" else "",
        })
    return jobs


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/clusters")
def api_clusters():
    clusters = get_clusters()
    result = {}
    for name, cfg in clusters.items():
        cp = _control_path(cfg)
        result[name] = {
            "label":        cfg.get("label", name),
            "host":         cfg["host"],
            "user":         cfg.get("user", ""),
            "notes":        cfg.get("notes", ""),
            "connected":    is_connected(cfg),
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
        return {"connected": False,
                "connect_cmd": f"ssh -M -S {_control_path(cfg)} -N -f {_ssh_target(cfg)}"}
    data = fetch_cluster_status(cfg)
    data["connected"] = True
    return data


@app.get("/api/clusters/{cluster_name}/myjobs")
def api_my_jobs(cluster_name: str):
    clusters = get_clusters()
    if cluster_name not in clusters:
        raise HTTPException(status_code=404, detail="Cluster not found")
    cfg = clusters[cluster_name]
    if not is_connected(cfg):
        raise HTTPException(status_code=503, detail="Not connected")
    return fetch_my_jobs(cfg)


@app.get("/api/clusters/{cluster_name}/debug")
def api_debug(cluster_name: str):
    """Returns raw sinfo output for diagnosing parsing issues."""
    clusters = get_clusters()
    if cluster_name not in clusters:
        raise HTTPException(status_code=404, detail="Cluster not found")
    cfg = clusters[cluster_name]
    if not is_connected(cfg):
        raise HTTPException(status_code=503, detail="Not connected")

    raw_sinfo, _, _ = run_remote(cfg, f"sinfo -N -h --Format={_SINFO_FORMAT} | head -20")
    raw_squeue, _, _ = run_remote(cfg, "squeue --me -h -o '%i|%j|%P|%T|%N|%b|%C|%m|%l|%S|%r'")
    return {"sinfo_sample": raw_sinfo, "squeue_me": raw_squeue,
            "sinfo_format": _SINFO_FORMAT}


@app.get("/api/clusters/{cluster_name}/disconnect")
def api_disconnect(cluster_name: str):
    clusters = get_clusters()
    if cluster_name not in clusters:
        raise HTTPException(status_code=404, detail="Cluster not found")
    cfg = clusters[cluster_name]
    subprocess.run(
        ["ssh", "-O", "exit", "-o", f"ControlPath={_control_path(cfg)}", _ssh_target(cfg)],
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
