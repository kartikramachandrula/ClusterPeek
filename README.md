# ClusterPeek

A lightweight web dashboard for monitoring SLURM compute clusters — GPU availability, node status, and your running jobs, all in one place.

## Features

- **GPU availability** — real-time counts by type (A100, H100, H200, L40S, etc.) across all partitions
- **Node status** — per-node GPU/CPU utilization and active users
- **Job tracking** — view your running and pending jobs with resource details
- **Multi-cluster** — monitor multiple SLURM clusters simultaneously
- **No stored credentials** — uses SSH ControlMaster for secure, password-free connections

## Setup

**Prerequisites:** Python 3.8+, SSH access to your cluster(s)

```bash
git clone https://github.com/your-username/ClusterPeek.git
cd ClusterPeek
python -m venv .venv && source .venv/bin/activate
python -m ensurepip   # only needed if pip3 is missing after activation
pip3 install -r requirements.txt
```

### Configure clusters

Edit `clusters.yaml` to add your cluster(s):

```yaml
clusters:
  my-cluster:
    host: cluster.example.edu
    user: your-username
```

### Connect via SSH ControlMaster

Run the included setup script to establish persistent SSH sessions for all configured clusters:

```bash
python connect.py
```

You'll be prompted to authenticate (Kerberos / Duo) for each cluster. Or connect manually:

```bash
ssh -M -S ~/.ssh/control-<host> -N -f your-username@cluster.example.edu
```

### Run

```bash
python main.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) in your browser.

## Usage

- **Partition Summary** — overview of GPU availability across all partitions
- **GPU Node Status** — per-node breakdown with users; search by username
- **My Jobs** — your current SLURM jobs; use **Request GPU** to submit a new allocation
