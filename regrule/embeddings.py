"""嵌入模型（Embedding）：把文字变成向量，语义相近的文字向量也相近。

三种来源，在 .env 里用 EMBEDDING_PROVIDER 选择：
- fastembed：本机运行的小模型 BAAI/bge-small-zh-v1.5（约 90MB，首次自动下载，不花钱）
- openai：任何兼容 OpenAI 格式的嵌入接口（如硅基流动、通义的 bge-m3 / text-embedding-v3）
- none：不用向量，只用 BM25 关键词检索

加载失败时自动退回 BM25，并在页面上提示原因，不会让整个程序挂掉。
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from .config import MODEL_CACHE, embedding_settings


class FastEmbedAdapter(Embeddings):
    """把 fastembed 包装成 LangChain 的 Embeddings 接口（只需实现两个方法）。"""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name, cache_dir=str(MODEL_CACHE))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


_override: Embeddings | None = None
_override_name = ""


def use_embeddings(model: Embeddings | None, name: str = "test") -> None:
    """测试时注入一个假的嵌入模型。"""
    global _override, _override_name
    _override, _override_name = model, name
    get_embeddings.cache_clear()


@lru_cache(maxsize=1)
def get_embeddings() -> tuple[Embeddings | None, str, str]:
    """返回 (嵌入模型或 None, 名称, 状态说明)。"""
    if _override is not None:
        return _override, _override_name, f"测试嵌入：{_override_name}"
    s = embedding_settings()
    try:
        if s.provider == "none":
            return None, "", "未启用向量检索（EMBEDDING_PROVIDER=none），只用 BM25。"
        if s.provider == "fastembed":
            return FastEmbedAdapter(s.model), s.model, f"本地嵌入模型：{s.model}"
        if s.provider == "openai":
            if not (s.base_url and s.api_key and s.model):
                return None, "", "嵌入接口未配置完整（EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL），只用 BM25。"
            from langchain_openai import OpenAIEmbeddings

            model = OpenAIEmbeddings(
                model=s.model, base_url=s.base_url, api_key=s.api_key,
                check_embedding_ctx_length=False,  # 非 OpenAI 官方接口需要关闭
            )
            return model, s.model, f"嵌入接口：{s.model}"
        return None, "", f"未知的 EMBEDDING_PROVIDER={s.provider}，只用 BM25。"
    except Exception as exc:  # 下载失败、网络不通等
        return None, "", f"嵌入模型加载失败，已退回 BM25：{type(exc).__name__}: {exc}"[:400]
