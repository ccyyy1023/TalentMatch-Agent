# JTH 基准接入说明

## 数据版本与许可

- 数据集：JTH - Job Tracking History
- Zenodo：<https://doi.org/10.5281/zenodo.21390581>
- 官方仓库：<https://github.com/Aunsiels/JTH>
- 许可：CC BY-NC 4.0，仅用于非商业研究与演示

原始数据位于 `data/external/jth/`，被 `.gitignore` 排除，项目不会二次分发 CSV。

| 文件 | 行数 | MD5 |
|---|---:|---|
| candidates.csv | 37,554 | 277c43a8f683726764a76dea1704be67 |
| jobs.csv | 6,011 | 3c586b3e87b95ed4efac3bcce1a68140 |
| history.csv | 42,288 | c2e0b7626f1c010d3eeb12567a7e37f2 |

## 用途与边界

JTH 没有原始简历和 JD 自由文本，因此只用于验证“一个岗位对历史申请者排序”的规模化能力，不能用于证明中文简历证据抽取效果。最后招聘阶段是历史行为弱标签，受到招聘流程、政策和既有偏差影响，不是真实岗位适配度金标。

系统不使用 `llm_sex`、`llm_nationality`、`llm_age_bucket`、薪资、邮编、来源、姓名标识等字段。匹配仅使用技能、岗位类别、专业领域、合同类型和经验区间等与岗位相关的结构化字段。

## 评测口径

- 测试岗位：`create_date >= 2024-01-01`。
- 候选池：该岗位的真实历史申请者，不把未申请者强行标成负样本。
- 最小候选池：5 人。
- 只保留至少有两个相关性等级且存在 `Resume Sent` 或更深阶段的岗位。
- 相关性等级：Application/Qualification=0，Shortlist=1，Resume Sent=2，Interview=3，Offer=4。
- Precision、Recall 和 MRR 将 `Resume Sent` 及更深阶段视为相关。
- 排序规则未用该测试集训练；当前权重为预先固定的透明规则。
- 以岗位查询为配对单位执行2,000次固定随机种子的bootstrap，报告各指标增量的95%置信区间。

运行：

```powershell
.\scripts\download_jth.ps1
.\.venv\Scripts\python.exe scripts\evaluate_jth.py
```

输出写入 `data/derived/jth_benchmark_report.json`。
