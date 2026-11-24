import os
import sys
import tempfile
from pathlib import Path

# 测试用独立的数据目录，不碰你自己的 .data
os.environ["REGRULE_DATA_DIR"] = tempfile.mkdtemp(prefix="regrule-test-")
os.environ["EMBEDDING_PROVIDER"] = "none"
os.environ.setdefault("LLM_API_KEY", "test-key-not-real")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
