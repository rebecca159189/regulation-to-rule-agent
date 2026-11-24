"""配置：全部从项目根目录的 .env 读取。

密钥只在本进程内使用，界面上只显示“已配置/未配置”和模型名，不显示密钥本身。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = Path(os.getenv("REGRULE_DATA_DIR", ROOT / ".data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
APP_DB = DATA_DIR / "app.sqlite"            # 文档、片段、任务、导出
CHECKPOINT_DB = DATA_DIR / "checkpoints.sqlite"  # LangGraph 的状态检查点
CHROMA_DIR = DATA_DIR / "chroma"            # 向量库
MODEL_CACHE = DATA_DIR / "models"           # 本地嵌入模型缓存
EXPORT_DIR = DATA_DIR / "exports"

# 限制（与旧版一致）
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_PDF_PAGES = 80
MAX_TEXT_CHARS = 120_000
MAX_CLAUSES = 300
MAX_RETRIEVALS = 30
MAX_CALLS_LIMIT = 6
DEFAULT_MAX_CALLS = 4


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    extra_body: dict
    tool_choice: str
    timeout: int

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


def llm_settings() -> LLMSettings:
    raw = _env("LLM_EXTRA_BODY", '{"thinking": {"type": "disabled"}}')
    try:
        extra = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        extra = {}
    return LLMSettings(
        base_url=_env("LLM_BASE_URL", "https://api.deepseek.com"),
        api_key=_env("LLM_API_KEY"),
        model=_env("LLM_MODEL", "deepseek-flash"),
        extra_body=extra,
        tool_choice=_env("LLM_TOOL_CHOICE", "required"),
        timeout=int(_env("LLM_TIMEOUT", "60") or 60),
    )


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str  # fastembed | openai | none
    model: str
    base_url: str
    api_key: str


def embedding_settings() -> EmbeddingSettings:
    provider = _env("EMBEDDING_PROVIDER", "fastembed").lower()
    default_model = {
        "fastembed": "BAAI/bge-small-zh-v1.5",
        "openai": "BAAI/bge-m3",
    }.get(provider, "")
    return EmbeddingSettings(
        provider=provider,
        model=_env("EMBEDDING_MODEL", default_model),
        base_url=_env("EMBEDDING_BASE_URL"),
        api_key=_env("EMBEDDING_API_KEY"),
    )
