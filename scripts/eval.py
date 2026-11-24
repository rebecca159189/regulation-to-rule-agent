"""评测脚本：把 10 份学习示例全跑一遍，统计 Agent 做对了几个。

用法（在项目根目录）：
    python scripts/eval.py --mode demo     # 演示模式，验证流程本身没坏（应 10/10）
    python scripts/eval.py --mode llm      # 真实大模型，看模型实际效果（会花一点 token 费用）

判定标准：
- 01~07：应该生成通过校验的规则，且金额条件的比较符、数值、单位都和标准答案一致；
- 08、10：原文缺信息，应该向人提问（ask_human），或者草稿里保留待确认问题，不能自己猜；
- 09：累计窗口当前模板做不了，应该标记 blocked，不能硬造单笔规则。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regrule import fixtures, service  # noqa: E402


def judge(example: dict, state: dict, pending: dict | None) -> tuple[bool, str]:
    behavior = fixtures.EXPECTED_BEHAVIOR.get(example["id"], "rule")
    draft, checks = state.get("draft") or {}, state.get("checks") or {}
    asked = bool(pending and pending["type"] == "clarify") or bool(draft.get("questions"))
    if behavior == "clarify":
        compiled = checks.get("compiled") and not checks.get("issues")
        return (asked and not compiled), ("提问了" if asked else "没提问") + ("，却生成了可导出规则" if compiled else "")
    if behavior == "blocked":
        blocked = any(c.get("disposition") == "blocked" for c in draft.get("coverage", []))
        compiled = bool(checks.get("compiled"))
        return (blocked and not compiled), f"blocked={blocked}，编译出规则={compiled}"
    if not checks or checks.get("issues"):
        codes = [i["code"] for i in checks.get("issues", [])] if checks else ["无草稿"]
        return False, f"校验未通过：{codes}"
    conds = [c for r in draft["rules"] for c in r["conditions"] if c["fieldId"] == "amount"]
    ok = any(c["operator"] == example["operator"] and c["value"] == example["value"] and c["unit"] == example["unit"]
             for c in conds)
    got = [(c["operator"], c["value"], c["unit"]) for c in conds]
    return ok, f"金额条件 {got}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "llm"], default="demo")
    parser.add_argument("--max-calls", type=int, default=4)
    args = parser.parse_args()

    passed = 0
    print(f"模式：{args.mode}\n")
    for ex in fixtures.EXAMPLES:
        ws = service.create_from_example(ex["id"])
        started = time.time()
        for _ in service.run(ws["id"], fixtures.DEFAULT_GOAL, args.mode, args.max_calls):
            pass
        state = service.snapshot(ws["id"]).values
        pending = service.pending_interrupt(ws["id"])
        ok, detail = judge(ex, state, pending)
        passed += ok
        tokens = sum(((c.get("usage") or {}).get("total_tokens") or 0) for c in state.get("call_log", []))
        print(f"{'✅' if ok else '❌'} {ex['name']:<16} 调用{state.get('llm_calls', 0)}次 "
              f"{tokens}tokens {time.time() - started:.1f}s  {detail}")
    print(f"\n通过 {passed}/{len(fixtures.EXAMPLES)}")


if __name__ == "__main__":
    main()
