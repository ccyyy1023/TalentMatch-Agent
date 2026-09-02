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

SENSITIVE_LINE_PATTERNS = (
    r"(?:性别|年龄|出生(?:日期|年月)?|婚姻(?:状况)?|民族|国籍|籍贯|政治面貌)\s*[:：]",
    r"(?:gender|sex|age|date\s+of\s+birth|marital\s+status|nationality|ethnicity|race)\s*[:：]",
    r"(?:^|\s)[男女](?:\s|$)",
    r"\b\d{1,2}\s*岁\b",
)


def detect_prompt_injection(text: str) -> list[str]:
    flags = []
    for index, pattern in enumerate(INJECTION_PATTERNS, start=1):
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            flags.append(f"prompt_injection_pattern_{index}")
    return flags


def sanitize_candidate_text(text: str, display_name: str = "") -> str:
    """Remove identity/contact-only lines before model extraction while preserving evidence lines verbatim."""
    retained = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if display_name and line == display_name.strip():
            continue
        if re.search(r"1[3-9]\d{9}|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", line):
            continue
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in SENSITIVE_LINE_PATTERNS):
            continue
        retained.append(line)
    return "\n".join(retained)
