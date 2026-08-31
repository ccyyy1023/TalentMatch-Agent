from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.schemas import AnalysisRequest  # noqa: E402
from app.services.workflow import TalentMatchWorkflow  # noqa: E402


def main() -> None:
    payload = json.loads((ROOT / "data" / "sample_dataset.json").read_text(encoding="utf-8"))
    payload["mode"] = "rules"
    original_request = AnalysisRequest.model_validate(payload)
    original = TalentMatchWorkflow().run(original_request)
    altered_payload = json.loads(json.dumps(payload, ensure_ascii=False))
    for candidate in altered_payload["candidates"]:
        candidate["name"] = "替换姓名"
        candidate["text"] = re.sub(r"\b(男|女)\b|\d{2}岁|出生年月[:：]?\S+", "", candidate["text"])
    altered = TalentMatchWorkflow().run(AnalysisRequest.model_validate(altered_payload))
    original_scores = {item.candidate_id: item.score for item in original.ranking}
    altered_scores = {item.candidate_id: item.score for item in altered.ranking}
    changes = {item_id: round(altered_scores[item_id] - score, 4) for item_id, score in original_scores.items()}
    report = {
        "test": "姓名替换并移除显式性别/年龄文本",
        "candidate_count": len(changes),
        "unchanged_count": sum(value == 0 for value in changes.values()),
        "ranking_unchanged": [item.candidate_id for item in original.ranking] == [item.candidate_id for item in altered.ranking],
        "score_changes": changes,
        "limitation": "该配对测试只能验证当前样本和显式属性，不等于证明系统无偏。",
    }
    output = ROOT / "data" / "reliability_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
