#!/usr/bin/env python3
"""GPU and LLM metrics for Prometheus, on one /metrics.

GPU: VRAM, busy time and clocks from sysfs, and VRAM per process from DRM fdinfo (the
container runs with pid: host). Temperatures, fans and power are left to node-exporter's
hwmon collector.

LLM: asks llama-swap which models are loaded and re-exports each one's /metrics
(llama-server --metrics, radiance) with model, backend and gpu labels. Models that are
not loaded are never contacted, so a scrape can't start one.

Config (env):
  GPU_PCI          r9700:0000:01:00.0,b50:0000:02:00.0   card name by PCI address
  GPUS             r9700:Vulkan0,b50:Vulkan1             card name by llama.cpp device
  GPU_VRAM_MIB     b50:16304                             VRAM size where sysfs has none (xe)
  RADIANCE_GPU     r9700
  LLAMA_SWAP       http://llama-swap:8080
  PORT             9835
"""
import glob
import json
import os
import re
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


def pairs(env):
    """'a:x,b:y' -> {'a': 'x', 'b': 'y'}; the value may hold colons (PCI addresses)."""
    out = {}
    for item in os.environ.get(env, "").split(","):
        name, sep, value = item.strip().partition(":")
        if sep and name and value:
            out[name] = value
    return out


GPU_BY_PCI = {pci: name for name, pci in pairs("GPU_PCI").items()}
GPU_BY_DEVICE = {dev: name for name, dev in pairs("GPUS").items()}
VRAM_MIB = {name: int(mib) for name, mib in pairs("GPU_VRAM_MIB").items()}
RADIANCE_GPU = os.environ.get("RADIANCE_GPU", "")
LLAMA_SWAP = os.environ.get("LLAMA_SWAP", "http://llama-swap:8080")
PORT = int(os.environ.get("PORT", "9835"))
TIMEOUT = 3

SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(\s+\S+)?$")
UNITS = {"": 1, "B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}


def esc(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def fmt(name, labels, value):
    inner = ",".join(f'{k}="{esc(v)}"' for k, v in labels.items())
    return f"{name}{{{inner}}} {value}"


def read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


# ------------------------------------------------------------------------- GPU

def cards():
    """(gpu, pci, driver, device dir) for every DRM card."""
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]")):
        dev = os.path.realpath(os.path.join(card, "device"))
        pci = os.path.basename(dev)
        driver = os.path.basename(os.path.realpath(os.path.join(dev, "driver")))
        yield GPU_BY_PCI.get(pci, pci), pci, driver, dev


def fdinfo_clients():
    """{(pdev, client id): (pid, comm, resident vram bytes)}: one entry per DRM client,
    so an fd shared by several processes or duplicated within one is counted once."""
    clients = {}
    for fd_dir in glob.glob("/proc/[0-9]*/fd"):
        pid = fd_dir.split("/")[2]
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                if not os.readlink(f"{fd_dir}/{fd}").startswith("/dev/dri/"):
                    continue
            except OSError:
                continue
            info = {}
            for line in (read(f"/proc/{pid}/fdinfo/{fd}", "") or "").splitlines():
                key, _, value = line.partition(":")
                info[key] = value.strip()
            if "drm-client-id" not in info or "drm-pdev" not in info:
                continue
            # amdgpu: drm-resident-vram, xe: drm-resident-vram0 (one per VRAM region)
            vram = 0
            for key, value in info.items():
                if re.fullmatch(r"drm-resident-vram\d*", key):
                    num, _, unit = value.partition(" ")
                    vram += int(num) * UNITS.get(unit, 1)
            key = (info["drm-pdev"], info["drm-client-id"])
            if key not in clients:
                clients[key] = (pid, read(f"/proc/{pid}/comm", "?"), vram)
    return clients


def gpu_metrics():
    out = []
    clients = fdinfo_clients()
    for gpu, pci, driver, dev in cards():
        base = {"gpu": gpu}
        out.append(fmt("ai_gpu_info", {**base, "pci": pci, "driver": driver}, 1))

        per_proc = {}
        for (pdev, _), (pid, comm, vram) in clients.items():
            if pdev == pci:
                per_proc[(pid, comm)] = per_proc.get((pid, comm), 0) + vram
        for (pid, comm), vram in per_proc.items():
            out.append(fmt("ai_gpu_process_vram_bytes", {**base, "pid": pid, "process": comm}, vram))

        used = read(f"{dev}/mem_info_vram_used")
        total = read(f"{dev}/mem_info_vram_total")
        if used is None:  # xe: no sysfs counter; sum what its clients hold
            used = sum(per_proc.values())
        if total is None and gpu in VRAM_MIB:
            total = VRAM_MIB[gpu] << 20
        out.append(fmt("ai_gpu_vram_used_bytes", base, used))
        if total is not None:
            out.append(fmt("ai_gpu_vram_total_bytes", base, total))

        busy = read(f"{dev}/gpu_busy_percent")  # amdgpu
        if busy is not None:
            out.append(fmt("ai_gpu_busy_percent", base, busy))
        mem_busy = read(f"{dev}/mem_busy_percent")
        if mem_busy is not None:
            out.append(fmt("ai_gpu_memory_busy_percent", base, mem_busy))

        # xe: per-GT idle residency (gt0-rc render/compute, gt1-mc media) and clock
        for gt in sorted(glob.glob(f"{dev}/tile*/gt*")):
            name = (read(f"{gt}/gtidle/name") or os.path.basename(gt)).split("-")[-1]
            idle_ms = read(f"{gt}/gtidle/idle_residency_ms")
            if idle_ms is not None:
                out.append(fmt("ai_gpu_gt_idle_seconds_total", {**base, "gt": name}, int(idle_ms) / 1000))
            freq = read(f"{gt}/freq0/act_freq")
            if freq is not None:
                out.append(fmt("ai_gpu_clock_mhz", {**base, "gt": name}, freq))
    return out


# ------------------------------------------------------------------------- LLM

def fetch(url):
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        return r.read().decode("utf-8", "replace")


def upstream_url(proxy):
    """llama-swap's proxy for a llama-server is 127.0.0.1:<port> inside its own container,
    where llama-server listens on all addresses: reach it by the llama-swap host instead."""
    p = urlparse(proxy)
    if p.hostname in ("127.0.0.1", "localhost"):
        return f"{p.scheme}://{urlparse(LLAMA_SWAP).hostname}:{p.port}"
    return proxy.rstrip("/")


def placement(model):
    cmd = model.get("cmd", "")
    if "radiance" in model.get("proxy", "") or " docker run " in f" {cmd} ":
        return "radiance", RADIANCE_GPU or "all"
    m = re.search(r"--device\s+(\S+)", cmd)
    if not m:
        return "llama", "all"
    names = [GPU_BY_DEVICE.get(d, d) for d in m.group(1).split(",")]
    return "llama", names[0] if len(names) == 1 else "all"


def relabel(text, extra):
    """Add extra labels to every sample. HELP/TYPE lines are dropped: the same family
    comes from several models, and a repeated TYPE line fails the whole scrape."""
    out = []
    add = ",".join(f'{k}="{esc(v)}"' for k, v in extra.items())
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = SAMPLE.match(line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2), m.group(3)
        out.append(f"{name}{{{add}{',' + labels if labels else ''}}} {value}")
    return out


def llm_metrics():
    out = []
    try:
        running = json.loads(fetch(f"{LLAMA_SWAP}/running")).get("running", [])
        out.append("ai_llm_swap_up 1")
    except Exception:
        return ["ai_llm_swap_up 0"]
    for model in running:
        backend, gpu = placement(model)
        labels = {"model": model.get("model", "?"), "backend": backend, "gpu": gpu}
        out.append(fmt("ai_llm_model_loaded", {**labels, "state": model.get("state", "?")}, 1))
        if model.get("state") != "ready":
            continue
        try:
            body = fetch(upstream_url(model.get("proxy", "")) + "/metrics")
            out.extend(relabel(body, labels))
            out.append(fmt("ai_llm_scrape_ok", labels, 1))
        except Exception:
            out.append(fmt("ai_llm_scrape_ok", labels, 0))
    return out


# ------------------------------------------------------------------------ HTTP

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] != "/metrics":
            self.send_error(404)
            return
        lines = []
        for collect in (gpu_metrics, llm_metrics):
            try:
                lines.extend(collect())
            except Exception as e:  # one broken source must not blank the other
                lines.append(fmt("ai_exporter_error", {"source": collect.__name__}, 1))
                print(f"{collect.__name__}: {e}", flush=True)
        body = ("\n".join(lines) + "\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
