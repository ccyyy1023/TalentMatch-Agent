# Third-party datasets and code

## JTH - Job Tracking History

- Source repository: https://github.com/Aunsiels/JTH
- Dataset record: https://doi.org/10.5281/zenodo.21390581
- License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
- Use in this project: offline, non-commercial research benchmark for job-to-candidate ranking.

The original CSV files are downloaded into `data/external/jth/` and are excluded
from version control. This repository does not redistribute the JTH dataset. Any
derived benchmark output must retain attribution and must not be represented as
commercial training data or as manually verified ground-truth job fit.

JTH interaction outcomes are behavioral weak labels affected by historical
recruiter decisions. Protected or irrelevant attributes such as inferred sex,
nationality, and age are excluded from matching features.

## TalentCLEF 2026 Task A

- Dataset record: https://zenodo.org/records/19652670
- Concept DOI: https://doi.org/10.5281/zenodo.17625261
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Use in this project: offline research benchmark for full-text contextual job-person ranking.

The English and Spanish job descriptions and CVs are synthetic,
privacy-preserving documents generated from structured real recruitment data
and manually reviewed by the dataset creators. The original archive and all
extracted documents remain under `data/external/talentclef2026/`, are excluded
from version control, and are not redistributed by this repository.

## SkillSpan

- Source repository: https://github.com/kris927b/SkillSpan
- Paper: https://aclanthology.org/2022.naacl-main.366
- Pinned data commit: `2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b`
- Repository license: MIT
- Use in this project: offline research benchmark for English job-posting skill and knowledge span extraction.

The released `house` and `tech` subsets are de-identified by the dataset
authors. Original JSON/CoNLL files remain under `data/external/skillspan/`, are
excluded from version control, and are not redistributed by this repository.

The optional JobBERT comparison uses the following Hugging Face model snapshots:

- `jjzha/jobbert_skill_extraction` at `a6211b675e287f3a07ffd4834b8182a2b519589b`
- `jjzha/jobbert_knowledge_extraction` at `9dea9759b7e290413cb4d92e8cae338affab9be4`

Both model cards cite the SkillSpan paper but currently have missing YAML
metadata and do not declare an artifact license. The weights are therefore
downloaded only for local evaluation, excluded from version control, and not
redistributed by this repository. The repository's MIT license is not assumed
to automatically license separately hosted model artifacts.
