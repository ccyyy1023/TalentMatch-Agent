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
