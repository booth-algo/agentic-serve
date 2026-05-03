import hashlib
import re


def resolve_model_name(model_path: str) -> str:
    """Convert '/workspace/models/meta-llama_Llama-3.1-8B-Instruct' -> 'Llama-3.1-8B-Instruct'."""
    name = model_path.rstrip("/").split("/")[-1]
    for prefix in ("meta-llama_", "Qwen_", "mistralai_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def detect_model_family(model: str) -> str:
    """'Llama-3.1-8B-Instruct' -> 'llama'."""
    lower = model.lower()
    if "llama" in lower:
        return "llama"
    if "qwen" in lower:
        return "qwen"
    if "gpt-oss" in lower:
        return "gpt-oss"
    if "mixtral" in lower:
        return "mixtral"
    if "gemma" in lower:
        return "gemma"
    if "granite" in lower:
        return "granite"
    return "unknown"


def parse_tensor_parallelism(hardware: str, config: dict) -> int:
    """Extract TP from config field or hardware name like 'H100x4'."""
    tp = config.get("tensor_parallel_size")
    if tp is not None:
        return int(tp)
    match = re.search(r"x(\d+)$", hardware)
    if match:
        return int(match.group(1))
    return 1


def generate_run_id(model: str, hardware: str, engine: str, tp: int, profile: str, concurrency: int) -> str:
    """Deterministic ID from run parameters."""
    payload = f"{model}|{hardware}|{engine}|{tp}|{profile}|{concurrency}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
