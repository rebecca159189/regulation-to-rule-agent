"""混合检索（Hybrid Retrieval）= 关键词检索 BM25 + 向量检索 + RRF 融合排序。

- BM25（rank_bm25 + jieba 中文分词）：擅长精确匹配“3000元”“名单A”这类字面词；
- 向量检索（Chroma 向量库 + 嵌入模型）：擅长语义相近，如“付款”和“转出”；
- RRF（倒数排名融合）：两路结果各自排名，分数 = Σ 1/(60+名次)，名次越靠前分越高。

检索严格分 3 个范围，互不串：
  document 本次文件 / platform 平台资料 / existing 已有规则
"""
from __future__ import annotations

import logging
import re
import threading
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)  # jieba 老代码的无害警告
import jieba  # noqa: E402
from rank_bm25 import BM25Okapi

from . import store
from .config import CHROMA_DIR
from .embeddings import get_embeddings
from .platform import existing_owner, platform_facts, platform_owner, reference_facts
from .schemas import Hit, Profile

jieba.setLogLevel(logging.WARNING)
for word in ("手机银行", "风险客户", "名单A", "人民币", "北京时间", "单笔", "付款", "累计"):
    jieba.add_word(word)

RRF_K = 60
_PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """中文分词：jieba 搜索引擎模式，去掉标点空白。"""
    return [t for t in (w.strip().lower() for w in jieba.lcut_for_search(text)) if t and not _PUNCT.match(t)]


# ---------------- 向量库 ----------------
_vs_lock = threading.Lock()
_vectorstores: dict[str, object] = {}
_indexed: set[tuple[str, str, str]] = set()


def _vectorstore():
    """按嵌入模型名分 collection（不同模型的向量维度不同，不能混放）。"""
    emb, name, _ = get_embeddings()
    if emb is None:
        return None, ""
    collection = "chunks_" + re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()[:50]
    with _vs_lock:
        if collection not in _vectorstores:
            import chromadb
            from chromadb.config import Settings
            from langchain_chroma import Chroma

            client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
            _vectorstores[collection] = Chroma(
                client=client, collection_name=collection, embedding_function=emb,
                collection_metadata={"hnsw:space": "cosine"},
            )
    return _vectorstores[collection], collection


def _where(owner: str, kind: str) -> dict:
    return {"$and": [{"owner": owner}, {"kind": kind}]}


def _ensure_vectors(owner: str, kind: str) -> None:
    """懒加载：某个范围第一次被检索时，才把它的片段写进向量库。"""
    vs, collection = _vectorstore()
    if vs is None or (collection, owner, kind) in _indexed:
        return
    rows = store.chunks_for(owner, kind)
    have = set(vs.get(where=_where(owner, kind), include=[])["ids"])
    todo = [r for r in rows if r["id"] not in have]
    if todo:
        vs.add_texts(
            texts=[f"{r['label']}\n{r['text']}" for r in todo],
            metadatas=[{"owner": r["owner"], "kind": r["kind"], "clause_id": r["clause_id"] or ""} for r in todo],
            ids=[r["id"] for r in todo],
        )
    _indexed.add((collection, owner, kind))


# ---------------- 写入 ----------------
def index_chunks(chunks: list[dict]) -> None:
    store.put_chunks(chunks)
    for owner, kind in {(c["owner"], c["kind"]) for c in chunks}:
        for key in [k for k in _indexed if k[1] == owner and k[2] == kind]:
            _indexed.discard(key)  # 有新片段，下次检索时补进向量库


def prepare_knowledge(profile: Profile) -> None:
    """把平台资料和已有参考规则写进检索库（已存在的会跳过）。"""
    store.put_chunks(platform_facts(profile))
    store.put_chunks(reference_facts(profile))


def owner_for(scope: str, document_id: str, profile: Profile) -> str:
    return {"document": document_id, "platform": platform_owner(profile), "existing": existing_owner(profile)}[scope]


# ---------------- 检索 ----------------
def search(query: str, scope: str, document_id: str, profile: Profile, k: int = 8) -> list[Hit]:
    if not isinstance(query, str) or not query.strip() or len(query) > 1000:
        raise ValueError("检索词需要 1~1000 个字符。")
    if scope not in ("document", "platform", "existing"):
        raise ValueError("检索范围只能是 document / platform / existing。")
    k = max(1, min(int(k), 15))
    prepare_knowledge(profile)
    owner = owner_for(scope, document_id, profile)
    rows = store.chunks_for(owner, scope)
    if not rows:
        return []
    by_id = {r["id"]: r for r in rows}

    # 按条款编号直接取原文
    if scope == "document" and query.strip().startswith(document_id + "-p"):
        q = query.strip()
        hits = [r for r in rows if r["id"] == q or r["clause_id"] == q][:k]
        return [Hit(**{**r, "score": 1.0}) for r in hits]

    # 第 1 路：BM25（词频 × 逆文档频率，再按文档长度归一）
    bm25 = BM25Okapi([tokenize(f"{r['label']} {r['text']}") for r in rows])
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens) if q_tokens else [0.0] * len(rows)
    bm25_ranked = [rows[i]["id"] for i in sorted(range(len(rows)), key=lambda i: -scores[i]) if scores[i] > 0][: k * 2]

    # 第 2 路：向量
    vector_ranked: list[str] = []
    vs, _ = _vectorstore()
    if vs is not None:
        try:
            _ensure_vectors(owner, scope)
            results = vs.similarity_search_with_score(query, k=min(k * 2, len(rows)), filter=_where(owner, scope))
            vector_ranked = [doc.id for doc, _ in results if doc.id in by_id]
        except Exception as exc:  # 向量库出问题不影响关键词检索
            logging.getLogger(__name__).warning("向量检索失败：%s", exc)

    # RRF 融合
    fused: dict[str, float] = {}
    for ranked in (bm25_ranked, vector_ranked):
        for rank, cid in enumerate(ranked, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    order = sorted(fused, key=lambda cid: (-fused[cid], cid))[:k]
    hits = []
    for cid in order:
        r = by_id[cid]
        hits.append(Hit(
            **r, score=round(fused[cid], 5),
            bm25_rank=bm25_ranked.index(cid) + 1 if cid in bm25_ranked else None,
            vector_rank=vector_ranked.index(cid) + 1 if cid in vector_ranked else None,
        ))
    return hits


def index_approved(bundle: dict, profile: Profile) -> None:
    """导出的规则加入“已有规则”范围，以后的任务能检索到。"""
    import json

    owner = existing_owner(profile)
    index_chunks([{
        "id": f"{owner}-{bundle['id']}-{r['key']}", "owner": owner, "kind": "existing",
        "label": f"{r['values']['rule_name']} · 本地已复核未启用草稿",
        "text": json.dumps({"source": bundle["document"]["name"], "rule": r,
                            "notice": "本地已复核草稿，不代表生产现有规则"}, ensure_ascii=False),
        "clause_id": None,
    } for r in bundle["nativeRules"]])
