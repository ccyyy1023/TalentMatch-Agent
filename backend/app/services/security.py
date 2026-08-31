from __future__ import annotations

import re


INJECTION_PATTERNS = (
    r"忽略(?:以上|上述|之前|系统|所有).{0,12}(?:指令|要求|规则)",
    r"(?:给我|将我|把我).{0,8}(?:100分|满分|排在第一|设为第一|优先推荐)",
    r"ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions?",
    r"(?:reveal|show|print).{0,12}system\s+prompt",
    r"你现在是.{0,20}(?:招聘官|系统|助手|模型)",
    r"system\s*prompt\s*[:：]",
)


def detect_prompt_injection(text: str) -> list[str]:
    flags = []
    for index, pattern in enumerate(INJECTION_PATTERNS, start=1):
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            flags.append(f"prompt_injection_pattern_{index}")
    return flags
