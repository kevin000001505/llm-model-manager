import glob
import json
import logging
import os
import re
import threading

import yaml

import docker as docker_sdk

from backend import config as cfg
from backend.config import (
    BACKEND_LLAMA, BACKEND_RADIANCE, CACHE_DIR, CONFIG_PATH, GPU_ALL, LLAMA_SHORT_FLAGS,
    LLAMA_SWAP_CONTAINER, QUANT_REGEX, RADIANCE_EXT, RADIANCE_LABEL, RADIANCE_PORT, SERVED_DIR,
)
from backend.state import iter_configs, load_state, save_state

log = logging.getLogger(__name__)

# radiance arguments go into a single-quoted `sh -c` string, so only plain tokens are allowed.
SAFE_ARG = re.compile(r"^[A-Za-z0-9_./:=,+@-]+$")


def stop_radiance_containers(client) -> None:
    """radiance containers are llama-swap's siblings, not its children: a llama-swap restart
    would leave them running and holding the card, so stop them first."""
    for container in client.containers.list(filters={"label": f"{RADIANCE_LABEL}=1"}):
        try:
            container.stop(timeout=60)
            log.info(f"Stopped radiance container {container.name}.")
        except Exception as e:
            log.error(f"Failed to stop radiance container {container.name}: {e}")


def restart_llama_swap() -> None:
    try:
        client = docker_sdk.from_env()
        stop_radiance_containers(client)
        container = client.containers.get(LLAMA_SWAP_CONTAINER)
        container.restart()
        log.info(f"Restarted {LLAMA_SWAP_CONTAINER} to apply new config.")
    except Exception as e:
        log.error(f"Failed to restart {LLAMA_SWAP_CONTAINER}: {e}")


def gpu_placement(data: dict) -> str:
    """The GPUS name a config runs on, or GPU_ALL. radiance always runs on RADIANCE_GPU; a name
    that has since left GPUS falls back to GPU_ALL."""
    if data.get("backend") == BACKEND_RADIANCE:
        gpu = cfg.RADIANCE_GPU or GPU_ALL
    else:
        gpu = data.get("gpu") or GPU_ALL
    if gpu != GPU_ALL and gpu not in cfg.GPUS:
        log.warning(f"GPU '{gpu}' is not in GPUS {list(cfg.GPUS)}; using all GPUs.")
        return GPU_ALL
    return gpu


def radiance_container_name(name: str) -> str:
    return "radiance-" + re.sub(r"[^A-Za-z0-9_.-]", "-", name)


def radiance_args(params: dict) -> list:
    """radiance flags from a config's params JSON. Unlike llama-server, every flag is long."""
    args = []
    for k, v in params.items():
        if k in ("host", "port", "model"):
            continue
        if v is None or (isinstance(v, bool) and v):
            args.append(f"--{k}")
        elif isinstance(v, bool) and not v:
            continue
        else:
            args.extend([f"--{k}", str(v)])
    bad = [a for a in args if not SAFE_ARG.match(a)]
    if bad:
        raise ValueError(f"radiance parameters may only use letters, digits and _./:=,+@-: {bad}")
    return args


def radiance_entry(name: str, params: dict) -> dict:
    """llama-swap model entry that runs radiance in its own container on the llama-swap network.
    llama-swap reaches it by container name, and stops it with cmdStop."""
    container = radiance_container_name(name)
    run = [
        "docker", "run", "--rm", "--init", "--name", container,
        "--label", f"{RADIANCE_LABEL}=1", "--network", cfg.LLAMA_NET,
        "--device", "/dev/kfd", "--device", "/dev/dri",
    ]
    for gid in cfg.RADIANCE_GROUPS:
        run.extend(["--group-add", gid])
    run.extend([
        # Docker's default seccomp profile refuses io_uring and the NUMA calls the ROCm runtime makes.
        "--security-opt", "seccomp=unconfined", "--ulimit", "memlock=-1:-1",
        "-v", f"{cfg.MODELS_HOST}:/models:ro", cfg.RADIANCE_IMAGE,
        "--model", f"/models/served/{name}{RADIANCE_EXT}",
        "--host", "0.0.0.0", "--port", str(RADIANCE_PORT),
    ])
    run.extend(radiance_args(params))
    bad = [a for a in run if not SAFE_ARG.match(a)]
    if bad:
        raise ValueError(f"unsafe characters in radiance command: {bad}")
    inner = " ".join(run)
    return {
        # A container left over from a llama-swap crash would block the name; remove it first.
        "cmd": f"sh -c 'docker rm -f {container} >/dev/null 2>&1; exec {inner} >/proc/1/fd/1 2>&1'",
        "cmdStop": f"docker stop -t 60 {container}",
        "proxy": f"http://{container}:{RADIANCE_PORT}",
        "unloadTimeout": 70,
    }


def matrix_routing(placements: dict):
    """llama-swap matrix routing for GPU placement: models on different GPUs run together, a new
    model on a busy GPU swaps out the one there, and a GPU_ALL model runs alone (it is in no set).
    None when fewer than two GPUs have models: llama-swap's default of one model at a time is the
    same thing then."""
    names, clauses = {}, []
    for gpu in cfg.GPUS:
        members = [model_id for model_id, g in placements.items() if g == gpu]
        if not members:
            continue
        refs = []
        for model_id in members:
            var = f"m{len(names)}"  # model IDs may hold characters the set expressions don't take
            names[var] = model_id
            refs.append(var)
        clauses.append("(" + " | ".join(refs) + ")")
    if len(clauses) < 2:
        return None
    return {"router": {"use": "matrix", "settings": {"matrix": {
        "vars": names,
        "sets": {"gpus": " & ".join(clauses)},
    }}}}


def sync_system(state: dict, restart: bool = True) -> None:
    """Rebuild served-dir symlinks and llama-swap config.yaml from state."""
    log.info("Syncing served dir and llama-swap config from state...")
    os.makedirs(SERVED_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(SERVED_DIR, "*.gguf")) + glob.glob(os.path.join(SERVED_DIR, f"*{RADIANCE_EXT}")):
        try:
            os.unlink(f)
        except Exception:
            pass

    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir(CACHE_DIR) if os.path.isdir(CACHE_DIR) else None
    rev_lookup = {}
    if cache is not None:
        for repo in cache.repos:
            for rev in repo.revisions:
                rev_lookup[(repo.repo_id, rev.commit_hash)] = rev

    swap_models = {}
    placements = {}
    has_radiance = False
    state_changed = False
    rpc_mode = state.get("_meta", {}).get("rpc_mode", False)

    for name, data in iter_configs(state):
        if data.get("status") in ("downloading", "error"):
            continue

        has_rpc = bool(data.get("params", {}).get("rpc"))
        if rpc_mode != has_rpc:
            continue

        repo = data.get("repo")
        sha = data.get("revision")
        quant = data.get("quant", "")
        mmproj = data.get("mmproj", "")
        params = data.get("params", {})
        backend = data.get("backend") or BACKEND_LLAMA

        rev = rev_lookup.get((repo, sha)) if repo and sha else None
        if rev is None:
            log.warning(f"Snapshot missing for {name} (repo={repo}, sha={(sha or '')[:12]}). Marking missing.")
            if data.get("status") != "missing":
                data["status"] = "missing"
                state_changed = True
            continue

        if backend == BACKEND_RADIANCE:
            rad_file = next((cf for cf in rev.files if cf.file_name == quant), None)
            if rad_file is None:
                log.warning(f"{quant} missing in snapshot for {name}. Marking missing.")
                if data.get("status") != "missing":
                    data["status"] = "missing"
                    state_changed = True
                continue
            if data.get("status") == "missing":
                data["status"] = "ready"
                state_changed = True
            if not cfg.MODELS_HOST:
                log.warning(f"Skipping radiance config {name}: MODELS_HOST is not set.")
                continue
            try:
                entry = radiance_entry(name, params)
            except ValueError as e:
                log.error(f"Skipping radiance config {name}: {e}")
                continue
            try:
                os.symlink(str(rad_file.file_path), os.path.join(SERVED_DIR, f"{name}{RADIANCE_EXT}"))
            except FileExistsError:
                pass
            swap_models[name] = entry
            placements[name] = gpu_placement(data)
            has_radiance = True
            continue

        quant_files = []
        mmproj_file = None
        for cf in rev.files:
            fname = cf.file_name
            match = re.search(QUANT_REGEX, fname, re.IGNORECASE)
            if not match:
                continue
            matched_q = match.group(1).upper()
            is_mmproj_file = "mmproj" in fname.lower()
            if is_mmproj_file:
                if mmproj and matched_q == mmproj.upper():
                    mmproj_file = cf
            elif matched_q == quant.upper():
                quant_files.append(cf)

        if not quant_files or (mmproj and mmproj_file is None):
            log.warning(f"Required files missing in snapshot for {name}. Marking missing.")
            if data.get("status") != "missing":
                data["status"] = "missing"
                state_changed = True
            continue

        if data.get("status") == "missing":
            data["status"] = "ready"
            state_changed = True

        quant_files.sort(key=lambda cf: cf.file_name)
        first_shard = None
        for cf in quant_files:
            shard_match = re.search(r'(-\d{5}-of-\d{5}\.gguf)$', cf.file_name, re.IGNORECASE)
            if shard_match:
                symlink_filename = f"{name}-{quant}{shard_match.group(1)}"
                if "-00001-of-" in shard_match.group(1):
                    first_shard = os.path.join(SERVED_DIR, symlink_filename)
            else:
                symlink_filename = f"{name}-{quant}.gguf"
                if first_shard is None:
                    first_shard = os.path.join(SERVED_DIR, symlink_filename)
            symlink_path = os.path.join(SERVED_DIR, symlink_filename)
            try:
                os.symlink(str(cf.file_path), symlink_path)
            except FileExistsError:
                pass

        model_path = first_shard

        cmd_args = ["llama-server", "--model", model_path, "--host", "0.0.0.0", "--port", "${PORT}"]

        if mmproj_file is not None:
            mmproj_path = os.path.join(SERVED_DIR, f"{name}-mmproj-{mmproj.upper()}.gguf")
            try:
                os.symlink(str(mmproj_file.file_path), mmproj_path)
            except FileExistsError:
                pass
            cmd_args.extend(["--mmproj", mmproj_path])

        gpu = gpu_placement(data)
        # Pin the config to its card(s), unless the params already pick devices themselves.
        if cfg.GPUS and not ({"dev", "device"} & params.keys()):
            devices = list(cfg.GPUS.values()) if gpu == GPU_ALL else [cfg.GPUS[gpu]]
            cmd_args.extend(["--device", ",".join(devices)])
        # Token counters on /metrics for the exporter (docker-compose.metrics.yml).
        if "metrics" not in params:
            cmd_args.append("--metrics")

        for k, v in params.items():
            if k in ("host", "port", "model", "mmproj"):
                continue
            flag = f"-{k}" if k in LLAMA_SHORT_FLAGS else f"--{k}"
            if v is None or (isinstance(v, bool) and v):
                cmd_args.append(flag)
            elif isinstance(v, bool) and not v:
                continue
            elif isinstance(v, (dict, list)):
                json_str = json.dumps(v, separators=(',', ':'))
                escaped = json_str.replace('\\', '\\\\').replace('"', '\\"')
                cmd_args.extend([flag, f'"{escaped}"'])
            else:
                cmd_args.extend([flag, str(v)])

        model_id = f"{name}-{quant}"
        inner_cmd = " ".join(cmd_args)
        wrapped_cmd = f"sh -c 'exec {inner_cmd} >/proc/1/fd/1 2>&1'"
        swap_models[model_id] = {
            "cmd": wrapped_cmd,
            "proxy": "http://127.0.0.1:${PORT}",
        }
        placements[model_id] = gpu

    if state_changed:
        save_state(state)

    swap_config = {}
    if has_radiance:
        # radiance loads, plans and warms its kernels before /health answers 200.
        swap_config["healthCheckTimeout"] = 900
    swap_config["models"] = swap_models
    routing = matrix_routing(placements) if cfg.GPUS else None
    if routing:
        swap_config["routing"] = routing

    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(swap_config, f, default_flow_style=False, sort_keys=False)
    log.info("llama-swap config.yaml rewritten.")

    if restart:
        threading.Thread(target=restart_llama_swap, daemon=True).start()
