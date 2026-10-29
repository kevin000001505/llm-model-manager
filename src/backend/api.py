import json
import logging
import os
import re
import shutil

from fastapi import BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect

from backend import config as cfg
from backend.cache import scan_cache, prune_unreferenced_revisions
from backend.config import BACKEND_RADIANCE, BACKENDS, GPU_ALL, QUANT_REGEX, RADIANCE_EXT
from backend.hf_hub import get_branches, get_commits, pre_flight_size, resolve_sha
from backend.models import ModelSetup, RevisionDeleteReq, RpcModeReq
from backend.state import format_bytes, iter_configs, load_state, save_state
from backend.sync import SAFE_ARG, radiance_args, sync_system
from backend.websocket import manager

log = logging.getLogger(__name__)


# --- WebSocket ---
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# --- API: Quants ---
async def get_quants(repo: str):
    from huggingface_hub import HfApi
    try:
        api = HfApi(token=None)
        log.info(f"Scanning repo '{repo}' for quant files...")
        info = api.model_info(repo, files_metadata=True)
        quants_dict = {}
        mmproj_dict = {}
        rads = []

        for f in info.siblings:
            if f.rfilename.endswith(RADIANCE_EXT):
                rads.append({"name": f.rfilename, "size_str": format_bytes(f.size or 0), "raw": f.size or 0})
                continue
            if not f.rfilename.endswith(".gguf"):
                continue
            match = re.search(QUANT_REGEX, f.rfilename, re.IGNORECASE)
            if not match:
                continue
            q_name = match.group(1).upper()
            q_size = f.size or 0
            if "mmproj" in f.rfilename.lower():
                mmproj_dict[q_name] = mmproj_dict.get(q_name, 0) + q_size
            else:
                quants_dict[q_name] = quants_dict.get(q_name, 0) + q_size

        quants_list = [{"name": q, "size_str": format_bytes(s), "raw": s} for q, s in quants_dict.items()]
        quants_list.sort(key=lambda x: x["raw"], reverse=True)
        mmproj_list = [{"name": q, "size_str": format_bytes(s), "raw": s} for q, s in mmproj_dict.items()]
        mmproj_list.sort(key=lambda x: x["raw"], reverse=True)

        rads.sort(key=lambda x: x["raw"], reverse=True)

        repo_name = repo.rsplit("/", 1)[-1] if "/" in repo else repo
        return {"quants": quants_list, "mmprojs": mmproj_list, "rads": rads, "repoName": repo_name}
    except Exception as e:
        log.error(f"Failed to fetch quants for {repo}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# --- API: Commits ---
async def api_get_commits(repo: str, revision: str = "main"):
    log.info(f"Fetching commits for repo: {repo}, revision: {revision}")
    try:
        state = load_state()
        pinned_shas = {
            d["revision"]
            for _, d in iter_configs(state)
            if d.get("repo") == repo and d.get("revision")
        }
        commits = get_commits(repo, revision=revision)
        for c in commits:
            c["pinned"] = c["sha"] in pinned_shas
        log.info(f"Got {len(commits)} commits for {repo}")
        return {"commits": commits}
    except Exception as e:
        log.error(f"Failed to fetch commits for {repo}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


# --- API: Branches ---
async def api_get_branches(repo: str):
    log.info(f"Fetching branches for repo: {repo}")
    try:
        branches = get_branches(repo)
        log.info(f"Got {len(branches)} branches for {repo}")
        return {"branches": branches}
    except Exception as e:
        log.error(f"Failed to fetch branches for {repo}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


# --- API: Local Models ---
async def get_local_models():
    cache = scan_cache()
    if cache is None:
        return {"models": []}

    state = load_state()
    used_by = {}
    for name, data in iter_configs(state):
        key = (data.get("repo"), data.get("revision"))
        if None in key:
            continue
        used_by.setdefault(key, []).append(name)

    models = []
    for repo in sorted(cache.repos, key=lambda r: r.repo_id):
        revisions = []
        for rev in sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True):
            files = []
            for cf in sorted(rev.files, key=lambda f: f.file_name):
                is_rad = cf.file_name.endswith(RADIANCE_EXT)
                if not is_rad and not cf.file_name.endswith(".gguf"):
                    continue
                match = None if is_rad else re.search(QUANT_REGEX, cf.file_name, re.IGNORECASE)
                files.append({
                    "name": cf.file_name,
                    "size_str": format_bytes(cf.size_on_disk),
                    "quant": "RAD" if is_rad else (match.group(1).upper() if match else None),
                    "is_mmproj": "mmproj" in cf.file_name.lower(),
                })
            if not files:
                continue
            revisions.append({
                "sha": rev.commit_hash,
                "short_sha": rev.commit_hash[:12],
                "size_str": format_bytes(rev.size_on_disk),
                "last_modified": rev.last_modified,
                "refs": sorted(rev.refs),
                "files": files,
                "used_by": used_by.get((repo.repo_id, rev.commit_hash), []),
            })
        if not revisions:
            continue
        models.append({
            "repo": repo.repo_id,
            "size_str": format_bytes(repo.size_on_disk),
            "revisions": revisions,
        })
    return {"models": models}


# --- API: GPUs ---
async def get_gpus():
    """The GPUs a config can be placed on, and whether the radiance backend is available."""
    return {
        "gpus": [{"name": name, "device": device} for name, device in cfg.GPUS.items()],
        "radiance": {
            "enabled": bool(cfg.MODELS_HOST),
            "gpu": cfg.RADIANCE_GPU if cfg.RADIANCE_GPU in cfg.GPUS else GPU_ALL,
        },
    }


def validate_placement(req: ModelSetup) -> None:
    """Normalise req.backend/req.gpu, raising 400 on a combination that can't be served."""
    if req.backend not in BACKENDS:
        raise HTTPException(status_code=400, detail=f"Unknown backend '{req.backend}' (expected one of {', '.join(BACKENDS)}).")
    if req.backend == BACKEND_RADIANCE:
        if not cfg.MODELS_HOST:
            raise HTTPException(status_code=400, detail="radiance backend is off: set MODELS_HOST (the host path of /models).")
        if not req.quant.endswith(RADIANCE_EXT):
            raise HTTPException(status_code=400, detail=f"radiance serves {RADIANCE_EXT} containers; '{req.quant}' is not one.")
        if not SAFE_ARG.match(req.symlink_name):
            raise HTTPException(status_code=400, detail="radiance config names may only use letters, digits and _./:=,+@-.")
        req.mmproj = ""
        req.gpu = cfg.RADIANCE_GPU if cfg.RADIANCE_GPU in cfg.GPUS else GPU_ALL
        return
    if req.quant.endswith(RADIANCE_EXT):
        raise HTTPException(status_code=400, detail=f"llama.cpp serves GGUF files; pick the radiance backend for {RADIANCE_EXT}.")
    if req.gpu != GPU_ALL and req.gpu not in cfg.GPUS:
        raise HTTPException(status_code=400, detail=f"Unknown GPU '{req.gpu}' (configured: {', '.join(cfg.GPUS) or 'none'}).")


# --- API: RPC Mode ---
async def toggle_rpc(req: RpcModeReq):
    state = load_state()
    state.setdefault("_meta", {})["rpc_mode"] = req.enabled
    save_state(state)
    sync_system(state)
    return {"status": f"RPC mode enabled: {req.enabled}"}


# --- API: Delete Revision ---
async def delete_revision(req: RevisionDeleteReq):
    state = load_state()
    blockers = [
        name for name, data in iter_configs(state)
        if data.get("repo") == req.repo and data.get("revision") == req.revision
    ]
    if blockers:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Revision is pinned by existing configs. Delete them first.",
                "used_by": blockers,
                "repo": req.repo,
                "revision": req.revision,
            },
        )

    cache = scan_cache()
    if cache is None:
        raise HTTPException(status_code=404, detail="Cache is empty.")

    known = any(
        rev.commit_hash == req.revision
        for repo in cache.repos for rev in repo.revisions
        if repo.repo_id == req.repo
    )
    if not known:
        raise HTTPException(status_code=404, detail="Revision not found in cache.")

    try:
        cache.delete_revisions(req.revision).execute()
    except Exception as e:
        log.error(f"delete_revisions failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {e}")

    sync_system(state)
    return {"status": "Deleted"}


# --- API: Setup Model ---
async def setup_model(req: ModelSetup, background_tasks: BackgroundTasks):
    from backend.download import process_model

    req.symlink_name = req.symlink_name.strip().replace("/", "-")
    req.quant = req.quant.strip()
    # An empty quant downloads every GGUF in the repo, and an empty name makes a config that
    # DELETE /api/configs/{name} can't reach, which pins its revision for good.
    if not req.symlink_name:
        raise HTTPException(status_code=400, detail="Config name (symlink name) is required.")
    if not req.quant:
        raise HTTPException(status_code=400, detail="Pick a quant (or .rad file) to download.")
    if req.original_name:
        req.original_name = req.original_name.replace("/", "-")
    log.info(f"Setup {req.symlink_name} ({req.quant}, rev={req.revision[:12] if req.revision != 'latest' else 'latest'})")

    try:
        params_dict = json.loads(req.parameters) if req.parameters.strip() else {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    validate_placement(req)
    if req.backend == BACKEND_RADIANCE:
        try:
            radiance_args(params_dict)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    try:
        ref = "main" if req.revision == "latest" else req.revision
        resolved_sha = resolve_sha(req.hf_repo, ref)
    except Exception as e:
        log.error(f"Could not resolve revision '{req.revision}' for {req.hf_repo}: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid repo/revision: {e}")
    req.revision = resolved_sha

    state = load_state()
    if req.original_name and req.original_name != req.symlink_name and req.original_name in state:
        del state[req.original_name]

    existing = state.get(req.symlink_name) or {}
    needs_download = (
        not existing
        or existing.get("repo") != req.hf_repo
        or existing.get("quant") != req.quant
        or existing.get("mmproj", "") != req.mmproj
        or existing.get("revision") != resolved_sha
        or existing.get("status") == "missing"
    )

    warning_msg = ""
    if needs_download:
        os.makedirs("/models/.cache", exist_ok=True)
        try:
            model_size = pre_flight_size(req.hf_repo, resolved_sha, req.quant, req.mmproj)
        except Exception as e:
            log.error(f"Pre-flight API check failed: {e}")
            raise HTTPException(status_code=400, detail=f"API Error: {e}")

        total, used, free = shutil.disk_usage("/models/.cache")
        if model_size > free:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient Storage. Model requires {format_bytes(model_size)}, but only {format_bytes(free)} is available.",
            )
        if (used + model_size) / total > 0.8:
            warning_msg = " (Warning: This will push disk usage above 80%)"

    state[req.symlink_name] = {
        "repo": req.hf_repo,
        "quant": req.quant,
        "mmproj": req.mmproj,
        "params": params_dict,
        "status": "downloading" if needs_download else "ready",
        "revision": resolved_sha,
        "backend": req.backend,
        "gpu": req.gpu,
    }
    save_state(state)

    if needs_download:
        background_tasks.add_task(process_model, req)
        return {"status": f"Provisioning started.{warning_msg}"}
    else:
        sync_system(state)
        return {"status": "Config updated!"}


# --- API: Delete Config ---
async def delete_config(symlink_name: str):
    log.info(f"Deleting config {symlink_name}")
    state = load_state()
    if symlink_name in state:
        del state[symlink_name]
        save_state(state)
        prune_unreferenced_revisions(state)
        sync_system(state)
    return {"status": "Config deleted!"}
