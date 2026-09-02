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

系统不使用 `llm_sex`、`llm_nationality`、`llm_age_bucket`、薪资、邮编、来源、姓名标识等字段。匹配只使用技能、岗位类别、专业领域、合同类型、经验区间、行业、软技能、语言、认证、职级和学历等与岗位相关的结构化字段。

`history.csv` 的一行是一次招聘阶段记录，同一岗位与候选人可能出现多行。评测先按唯一岗位—候选人关系折叠，并保留该关系的最终阶段标签。2024+ 固定基准最终包含365个岗位、7,850个唯一人岗对和8,297条阶段记录，不能把阶段记录数直接写成人岗对数。

## 评测口径

- 测试岗位：`create_date >= 2024-01-01`。
- 权重搜索：2022年岗位；配置选择：2023年岗位；2024年及以后岗位仅在方案冻结后运行一次最终评测。
- 候选池：该岗位的真实历史申请者，不把未申请者强行标成负样本。
- 最小候选池：5 人。
- 只保留至少有两个相关性等级且存在 `Resume Sent` 或更深阶段的岗位。
- 相关性等级：Application/Qualification=0，Shortlist=1，Resume Sent=2，Interview=3，Offer=4。
- Precision、Recall 和 MRR 将 `Resume Sent` 及更深阶段视为相关。
- 同时报告三种方法：只用原始技能列的关键词基线、只用原始非LLM字段的多属性强基线，以及使用扩展结构化字段的可解释加权匹配器。
- 权重带有技能和岗位属性占比边界，不允许无约束拟合弱标签；每个字段权重及测试结果均写入JSON报告。
- 以岗位查询为配对单位执行2,000次固定随机种子的bootstrap，报告各指标增量的95%置信区间。

运行：

```powershell
.\scripts\download_jth.ps1
.\.venv\Scripts\python.exe scripts\evaluate_jth.py
```

输出写入 `data/derived/jth_benchmark_report.json`。

## 锚定式 LambdaMART 实验

在固定加权匹配器之外，项目提供一个离线学习排序候选方案：

- 只使用29个非敏感、具有业务含义的匹配特征，并对“覆盖率/匹配度应增分、缺口应减分”施加单调约束；
- 使用2019—2022训练、2023验证，以及2019—2023训练、2024验证的两折滚动时间验证，避免随机切分把未来招聘模式泄漏到过去；
- 学习分只占最终排序信号的10%，其余90%来自固定结构化分数的岗位内次序，控制弱标签漂移风险；
- 所有配置锁定后，只在2025年留出集上运行一次最终评测。

2025留出集包含49个有效岗位、1,012个唯一人岗对。固定规则与锚定式LambdaMART结果如下：

| 方法 | NDCG@5 | Precision@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| 固定结构化匹配 | 0.5010 | 0.2857 | 0.5091 | 0.5123 |
| 锚定式LambdaMART | 0.5156 | 0.2857 | 0.5295 | 0.5333 |

NDCG@5增量为0.0146，按岗位配对bootstrap的95%置信区间为[0.0005, 0.0323]；49个岗位中5个提升、44个持平、0个下降。它通过了离线候选门槛，但留出岗位数较少，且JTH为法国招聘历史弱标签。因此当前结论是“进入真实企业数据影子评测”，不是直接替换线上默认规则。

复现命令：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ranking.txt
.\.venv\Scripts\python.exe scripts\evaluate_jth_lambdamart.py
```

完整搜索报告与模型写入被版本控制排除的`data/derived/`；可公开的核心结果保存在`data/jth_lambdamart_summary.json`。
