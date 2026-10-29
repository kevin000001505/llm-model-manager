import os
from unittest.mock import MagicMock, patch

import pytest
import yaml
from fastapi import HTTPException

from backend.api import get_gpus, get_quants, validate_placement
from backend.config import parse_gpus
from backend.hf_hub import pre_flight_size
from backend.models import ModelSetup
from backend.sync import radiance_args, restart_llama_swap, sync_system

GPUS = {"r9700": "Vulkan0", "b50": "Vulkan1"}


def cache_with(*repos):
    """A scan_cache_dir result: repos given as (repo_id, sha, [file names])."""
    cache = MagicMock()
    cache.repos = []
    for repo_id, sha, names in repos:
        repo = MagicMock(repo_id=repo_id)
        rev = MagicMock(commit_hash=sha)
        rev.files = []
        for n in names:
            f = MagicMock(file_path=f"/models/.cache/{repo_id}/{n}")
            f.file_name = n
            rev.files.append(f)
        repo.revisions = [rev]
        cache.repos.append(repo)
    return cache


def gguf(name, gpu="all", params=None, quant="Q4_K_M"):
    return name, {
        "repo": f"test/{name}", "quant": quant, "mmproj": "", "params": params or {},
        "status": "ready", "revision": f"sha-{name}", "backend": "llama", "gpu": gpu,
    }


def rad(name, params=None):
    return name, {
        "repo": f"test/{name}", "quant": f"{name}.rad", "mmproj": "", "params": params or {},
        "status": "ready", "revision": f"sha-{name}", "backend": "radiance", "gpu": "r9700",
    }


def run_sync(tmp_state_dir, configs, gpus=GPUS, models_host="/opt/models", radiance_gpu="r9700"):
    """sync_system over configs (name, data) pairs; returns the written config.yaml."""
    os.makedirs(tmp_state_dir / ".cache", exist_ok=True)
    state = dict(configs)
    repos = []
    for name, data in configs:
        fname = data["quant"] if data["backend"] == "radiance" else f"model-{data['quant']}.gguf"
        repos.append((data["repo"], data["revision"], [fname]))
    with patch("huggingface_hub.scan_cache_dir", return_value=cache_with(*repos)), \
            patch("backend.config.GPUS", gpus), \
            patch("backend.config.MODELS_HOST", models_host), \
            patch("backend.config.RADIANCE_GPU", radiance_gpu), \
            patch("backend.config.RADIANCE_GROUPS", ["992", "44"]):
        sync_system(state, restart=False)
    with open(tmp_state_dir / "config.yaml") as f:
        return yaml.safe_load(f), state


class TestParseGpus:
    def test_parses_in_order(self):
        assert list(parse_gpus("r9700:Vulkan0, b50:Vulkan1").items()) == [("r9700", "Vulkan0"), ("b50", "Vulkan1")]

    def test_empty_is_no_gpus(self):
        assert parse_gpus("") == {}

    @pytest.mark.parametrize("spec", ["r9700", ":Vulkan0", "all:Vulkan0"])
    def test_rejects_bad_entries(self, spec):
        with pytest.raises(ValueError):
            parse_gpus(spec)


class TestLlamaPlacement:
    def test_pins_config_to_its_gpu(self, tmp_state_dir):
        cfg, _ = run_sync(tmp_state_dir, [gguf("small", gpu="b50")])
        assert "--device Vulkan1 " in cfg["models"]["small-Q4_K_M"]["cmd"]

    def test_all_gpus_lists_every_device(self, tmp_state_dir):
        cfg, _ = run_sync(tmp_state_dir, [gguf("big")])
        assert "--device Vulkan0,Vulkan1 " in cfg["models"]["big-Q4_K_M"]["cmd"]

    def test_params_device_wins(self, tmp_state_dir):
        cfg, _ = run_sync(tmp_state_dir, [gguf("own", gpu="b50", params={"dev": "Vulkan0"})])
        cmd = cfg["models"]["own-Q4_K_M"]["cmd"]
        assert "--device" not in cmd
        assert "-dev Vulkan0" in cmd

    def test_no_gpus_keeps_old_config(self, tmp_state_dir):
        cfg, _ = run_sync(tmp_state_dir, [gguf("a", gpu="b50"), gguf("b")], gpus={})
        assert "--device" not in cfg["models"]["a-Q4_K_M"]["cmd"]
        assert "routing" not in cfg

    def test_unknown_gpu_falls_back_to_all(self, tmp_state_dir):
        cfg, _ = run_sync(tmp_state_dir, [gguf("gone", gpu="old-card")])
        assert "--device Vulkan0,Vulkan1 " in cfg["models"]["gone-Q4_K_M"]["cmd"]


class TestMatrixRouting:
    def test_one_model_per_gpu_and_all_alone(self, tmp_state_dir):
        cfg, _ = run_sync(tmp_state_dir, [
            gguf("r1", gpu="r9700"), gguf("r2", gpu="r9700"), gguf("b1", gpu="b50"), gguf("both"),
        ])
        matrix = cfg["routing"]["router"]["settings"]["matrix"]
        assert cfg["routing"]["router"]["use"] == "matrix"
        assert matrix["vars"] == {"m0": "r1-Q4_K_M", "m1": "r2-Q4_K_M", "m2": "b1-Q4_K_M"}
        assert matrix["sets"] == {"gpus": "(m0 | m1) & (m2)"}
        assert "both-Q4_K_M" not in matrix["vars"].values()

    def test_no_routing_when_one_gpu_used(self, tmp_state_dir):
        cfg, _ = run_sync(tmp_state_dir, [gguf("r1", gpu="r9700"), gguf("both")])
        assert "routing" not in cfg


class TestRadiance:
    def test_writes_container_entry(self, tmp_state_dir):
        cfg, _ = run_sync(tmp_state_dir, [rad("qwen27", params={"tp": 1, "kv-cache-dtype": "fp8"})])
        entry = cfg["models"]["qwen27"]
        cmd = entry["cmd"]
        assert cmd.startswith("sh -c 'docker rm -f radiance-qwen27 >/dev/null 2>&1; exec docker run ")
        for part in ("--name radiance-qwen27", "--network llm-llama-net", "--device /dev/kfd",
                     "--group-add 992 --group-add 44", "-v /opt/models:/models:ro",
                     "--model /models/served/qwen27.rad", "--port 8000", "--tp 1 --kv-cache-dtype fp8"):
            assert part in cmd
        assert entry["cmdStop"] == "docker stop -t 60 radiance-qwen27"
        assert entry["proxy"] == "http://radiance-qwen27:8000"
        assert cfg["healthCheckTimeout"] == 900
        assert os.path.islink(tmp_state_dir / "qwen27.rad")

    def test_shares_r9700_with_llama_configs(self, tmp_state_dir):
        cfg, _ = run_sync(tmp_state_dir, [rad("qwen27"), gguf("r1", gpu="r9700"), gguf("b1", gpu="b50")])
        matrix = cfg["routing"]["router"]["settings"]["matrix"]
        assert matrix["vars"] == {"m0": "qwen27", "m1": "r1-Q4_K_M", "m2": "b1-Q4_K_M"}
        assert matrix["sets"] == {"gpus": "(m0 | m1) & (m2)"}

    def test_skipped_without_models_host(self, tmp_state_dir):
        cfg, _ = run_sync(tmp_state_dir, [rad("qwen27")], models_host="")
        assert cfg["models"] == {}
        assert "healthCheckTimeout" not in cfg

    def test_missing_rad_file_marks_missing(self, tmp_state_dir):
        name, data = rad("qwen27")
        os.makedirs(tmp_state_dir / ".cache", exist_ok=True)
        state = {name: data}
        cache = cache_with((data["repo"], data["revision"], ["README.md"]))
        with patch("huggingface_hub.scan_cache_dir", return_value=cache), \
                patch("backend.config.MODELS_HOST", "/opt/models"):
            sync_system(state, restart=False)
        assert state[name]["status"] == "missing"

    def test_rejects_unsafe_params(self):
        with pytest.raises(ValueError):
            radiance_args({"chat-template": "a b'; rm -rf /"})

    def test_bool_params_are_flags(self):
        assert radiance_args({"deterministic": True, "debug-graph": False, "tp": 2}) == ["--deterministic", "--tp", "2"]

    @patch("backend.sync.docker_sdk")
    def test_restart_stops_radiance_first(self, mock_docker):
        client = MagicMock()
        running = MagicMock()
        client.containers.list.return_value = [running]
        mock_docker.from_env.return_value = client
        restart_llama_swap()
        client.containers.list.assert_called_once_with(filters={"label": "llm-model-manager.radiance=1"})
        running.stop.assert_called_once_with(timeout=60)
        client.containers.get.return_value.restart.assert_called_once()


class TestRadianceFiles:
    @patch("backend.hf_hub.HfApi")
    def test_pre_flight_size_counts_the_rad_file(self, mock_api_class):
        files = []
        for name, size in (("q.rad", 19_000), ("README.md", 4), ("other.rad", 7)):
            f = MagicMock(size=size)
            f.rfilename = name
            files.append(f)
        mock_api_class.return_value.model_info.return_value = MagicMock(siblings=files)
        assert pre_flight_size("x/y", "sha", "q.rad", "") == 19_000

    @patch("huggingface_hub.HfApi")
    @pytest.mark.asyncio
    async def test_get_quants_lists_rads(self, mock_api_class):
        f = MagicMock(size=19_000_000_000)
        f.rfilename = "qwen3.8-27b-mxfp4.rad"
        mock_api_class.return_value.model_info.return_value = MagicMock(siblings=[f])
        result = await get_quants("StillDeadcode/qwen3.8-27b-mxfp4")
        assert result["quants"] == []
        assert [r["name"] for r in result["rads"]] == ["qwen3.8-27b-mxfp4.rad"]


def setup_req(**kw):
    base = dict(hf_repo="a/b", quant="Q4_K_M", symlink_name="m", parameters="{}")
    base.update(kw)
    return ModelSetup(**base)


class TestValidatePlacement:
    @pytest.fixture(autouse=True)
    def gpus(self):
        with patch("backend.config.GPUS", GPUS), patch("backend.config.RADIANCE_GPU", "r9700"), \
                patch("backend.config.MODELS_HOST", "/opt/models"):
            yield

    def test_radiance_forced_onto_its_gpu(self):
        req = setup_req(backend="radiance", quant="q.rad", gpu="b50", mmproj="F16")
        validate_placement(req)
        assert (req.gpu, req.mmproj) == ("r9700", "")

    @pytest.mark.parametrize("kw", [
        dict(backend="vllm"),
        dict(backend="radiance", quant="Q4_K_M"),
        dict(backend="llama", quant="q.rad"),
        dict(gpu="rtx"),
        dict(backend="radiance", quant="q.rad", symlink_name="has space"),
    ])
    def test_rejects(self, kw):
        with pytest.raises(HTTPException) as e:
            validate_placement(setup_req(**kw))
        assert e.value.status_code == 400

    def test_radiance_off_without_models_host(self):
        with patch("backend.config.MODELS_HOST", ""):
            with pytest.raises(HTTPException):
                validate_placement(setup_req(backend="radiance", quant="q.rad"))

    @pytest.mark.asyncio
    async def test_get_gpus(self):
        assert await get_gpus() == {
            "gpus": [{"name": "r9700", "device": "Vulkan0"}, {"name": "b50", "device": "Vulkan1"}],
            "radiance": {"enabled": True, "gpu": "r9700"},
        }
