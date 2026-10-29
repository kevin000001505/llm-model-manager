import os
import re

# --- Paths ---
CACHE_DIR = "/models/.cache"
SERVED_DIR = "/models/served"
CONFIG_PATH = os.path.join(SERVED_DIR, "config.yaml")
STATE_FILE = os.path.join(SERVED_DIR, "state.json")
LLAMA_SWAP_CONTAINER = os.environ.get("LLAMA_SWAP_CONTAINER", "llama-swap")


# --- GPUs ---
def parse_gpus(spec: str) -> dict:
    """'r9700:Vulkan0,b50:Vulkan1' -> {'r9700': 'Vulkan0', 'b50': 'Vulkan1'}, in order.
    The value is the llama.cpp device name (`llama-server --list-devices`)."""
    gpus = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, device = item.partition(":")
        name, device = name.strip(), device.strip()
        if not name or not device or name == GPU_ALL:
            raise ValueError(f"GPUS entry {item!r} must be name:device, e.g. r9700:Vulkan0")
        gpus[name] = device
    return gpus


# A config placed on GPU_ALL spans every card and runs alone. With GPUS unset, every config is
# GPU_ALL and llama-swap keeps its default of one model at a time.
GPU_ALL = "all"
GPUS = parse_gpus(os.environ.get("GPUS", ""))

# --- Backends ---
BACKEND_LLAMA = "llama"
BACKEND_RADIANCE = "radiance"
BACKENDS = (BACKEND_LLAMA, BACKEND_RADIANCE)

# radiance runs as a sibling container that llama-swap starts through the docker socket, so its
# volume must name the HOST path of /models. Leaving MODELS_HOST unset turns the backend off.
MODELS_HOST = os.environ.get("MODELS_HOST", "")
RADIANCE_IMAGE = os.environ.get("RADIANCE_IMAGE", "stilldeadcode/radiance:latest")
RADIANCE_GPU = os.environ.get("RADIANCE_GPU", "")  # the GPUS entry radiance runs on
RADIANCE_GROUPS = [g.strip() for g in os.environ.get("RADIANCE_GROUPS", "").split(",") if g.strip()]  # host render/video GIDs
RADIANCE_PORT = 8000
RADIANCE_LABEL = "llm-model-manager.radiance"
LLAMA_NET = os.environ.get("LLAMA_NET", "llm-llama-net")
RADIANCE_EXT = ".rad"

# --- Patterns ---
QUANT_REGEX = r'[-._]((?:UD-)?[A-Za-z]*Q[0-9][A-Za-z0-9_]*|BF16|F16|F32|MXFP4_MOE)(?:-\d{5}-of-\d{5})?\.gguf$'
SHA_RE = re.compile(r'/snapshots/([a-f0-9]{40})/')

# llama-server short flags (single dash) — everything else uses double dash
LLAMA_SHORT_FLAGS = {
    "a", "b", "bs", "c", "cb", "cd", "cl", "cmoe", "cmoed", "cpent", "cram",
    "ctk", "ctkd", "ctv", "ctvd", "ctxcp", "dev", "devd", "dio", "dr", "dt",
    "e", "fa", "fit", "fitc", "fitt", "h", "hf", "hfd", "hff", "hffv", "hft",
    "hfv", "j", "jf", "kvo", "kvu", "l", "lcd", "lcs", "lv", "m", "md", "mg",
    "mm", "mmu", "mu", "mv", "n", "ncmoe", "ngl", "ngld", "np", "ot", "otd",
    "r", "rea", "s", "sm", "sp", "sps", "t", "tb", "tbd", "td", "to", "ts",
    "ub", "v",
}
