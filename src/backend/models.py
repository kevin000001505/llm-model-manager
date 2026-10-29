from pydantic import BaseModel


class ModelSetup(BaseModel):
    hf_repo: str
    quant: str
    mmproj: str = ""
    symlink_name: str
    original_name: str = ""
    parameters: str
    revision: str = "latest"
    backend: str = "llama"  # "llama" (GGUF via llama-server) or "radiance" (.rad container)
    gpu: str = "all"  # a GPUS name, or "all"


class RevisionDeleteReq(BaseModel):
    repo: str
    revision: str


class RpcModeReq(BaseModel):
    enabled: bool
