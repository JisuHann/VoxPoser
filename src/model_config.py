"""
LLM 모델 설정 파일
Hugging Face 텍스트 기반 LLM 모델 설정
"""

# Hugging Face 텍스트 기반 LLM 모델 설정
HF_LLM_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "device": "cuda",  # "cuda" or "cpu"
    "torch_dtype": "float16",  # "float16" or "float32"
    "trust_remote_code": True,
    "device_map": "auto",  # "auto" or None
    "max_memory": None,  # GPU 메모리 제한 (예: {0: "10GB", 1: "10GB"})
    "low_cpu_mem_usage": True,
    "use_cache": True
}

# LMP 설정에서 사용할 기본값들
LMP_DEFAULT_CONFIG = {
    "temperature": 0.0,
    "max_tokens": 128,
    "stop": ["</s>", "```"],
    "load_cache": True,
    "maintain_session": True,
    "include_context": False,
    "has_return": False,
    "debug": True
}

def get_model_config():
    """모델 설정 반환"""
    return HF_LLM_CONFIG

def get_lmp_config():
    """LMP 기본 설정 반환"""
    return LMP_DEFAULT_CONFIG
