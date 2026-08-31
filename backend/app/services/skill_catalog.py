from __future__ import annotations

import re


SKILL_ALIASES: dict[str, set[str]] = {
    "python": {"python", "py"},
    "sql": {"sql", "结构化查询语言"},
    "fastapi": {"fastapi"},
    "flask": {"flask"},
    "django": {"django"},
    "pytorch": {"pytorch", "torch"},
    "tensorflow": {"tensorflow", "tf"},
    "machine_learning": {"机器学习", "machine learning", "ml"},
    "deep_learning": {"深度学习", "deep learning", "dl"},
    "nlp": {"自然语言处理", "nlp", "natural language processing"},
    "llm": {"大语言模型", "大模型", "llm", "large language model"},
    "rag": {"rag", "检索增强生成", "retrieval augmented generation"},
    "langchain": {"langchain"},
    "langgraph": {"langgraph"},
    "ollama": {"ollama"},
    "vllm": {"vllm"},
    "docker": {"docker", "容器化"},
    "kubernetes": {"kubernetes", "k8s"},
    "git": {"git", "github", "gitlab"},
    "linux": {"linux"},
    "redis": {"redis"},
    "postgresql": {"postgresql", "postgres", "pgvector"},
    "mysql": {"mysql"},
    "mongodb": {"mongodb", "mongo"},
    "elasticsearch": {"elasticsearch", "es"},
    "faiss": {"faiss"},
    "milvus": {"milvus"},
    "qdrant": {"qdrant"},
    "java": {"java"},
    "spring_boot": {"spring boot", "springboot"},
    "javascript": {"javascript", "js"},
    "typescript": {"typescript", "ts"},
    "react": {"react", "reactjs"},
    "vue": {"vue", "vue.js", "vuejs"},
    "fast_api": {"fast api"},
    "rest_api": {"rest api", "restful", "接口开发", "api开发"},
    "microservices": {"微服务", "microservice", "microservices"},
    "data_analysis": {"数据分析", "data analysis", "pandas"},
    "spark": {"spark", "pyspark"},
    "hadoop": {"hadoop"},
    "airflow": {"airflow"},
    "aws": {"aws", "amazon web services"},
    "azure": {"azure"},
    "gcp": {"gcp", "google cloud"},
}

DISPLAY_NAMES = {
    "machine_learning": "机器学习",
    "deep_learning": "深度学习",
    "nlp": "NLP",
    "llm": "大语言模型",
    "rag": "RAG",
    "spring_boot": "Spring Boot",
    "rest_api": "REST API",
    "microservices": "微服务",
    "data_analysis": "数据分析",
}


def display_name(skill: str) -> str:
    return DISPLAY_NAMES.get(skill, skill.replace("_", " ").title())


def normalize_skill(value: str) -> str | None:
    lowered = value.strip().lower()
    for canonical, aliases in SKILL_ALIASES.items():
        if lowered == canonical or lowered in aliases:
            return canonical
    return None


def extract_skills(text: str) -> list[tuple[str, str]]:
    lowered = text.lower()
    found: list[tuple[str, str]] = []
    for canonical, aliases in SKILL_ALIASES.items():
        matched_alias = None
        for alias in sorted(aliases, key=len, reverse=True):
            pattern = rf"(?<![a-z0-9_]){re.escape(alias.lower())}(?![a-z0-9_])" if alias.isascii() else re.escape(alias)
            for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
                prefix = lowered[max(0, match.start() - 12):match.start()]
                clause_start = max(lowered.rfind(mark, 0, match.start()) for mark in "，,。；;\n") + 1
                clause_prefix = lowered[clause_start:match.start()]
                directly_negated = re.search(
                    r"(?:未使用|未接触|不会|不熟悉|不具备|没有|无需|无)(?:过|任何)?[\s、,，/]*$",
                    prefix,
                )
                negated_clause = re.search(
                    r"(?:未使用|未接触|不会|不熟悉|不具备|没有|无需|无)(?:过|任何)?",
                    clause_prefix,
                )
                if not directly_negated and not negated_clause:
                    matched_alias = alias
                    break
            if matched_alias:
                break
        if matched_alias:
            found.append((canonical, matched_alias))
    return found


def lexical_relatedness(required: str, candidate: str) -> float:
    if required == candidate:
        return 1.0
    related_groups = [
        {"pytorch", "tensorflow", "deep_learning"},
        {"langchain", "langgraph", "rag", "llm"},
        {"postgresql", "mysql", "sql"},
        {"faiss", "milvus", "qdrant", "elasticsearch"},
        {"fastapi", "flask", "django", "rest_api"},
        {"react", "vue", "javascript", "typescript"},
    ]
    for group in related_groups:
        if required in group and candidate in group:
            return 0.45
    return 0.0
