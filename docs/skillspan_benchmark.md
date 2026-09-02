# SkillSpan技能Span抽取评测

## 目的与边界

SkillSpan层只回答一个问题：系统或模型能否在真实英文岗位句子中定位明确的技能与知识短语。它不评估候选人排序、招聘决策、公平性或企业生产效果，也不与JTH、TalentCLEF指标混写。

数据来自SkillSpan官方仓库，固定到commit `2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b`。本地释放文件包含train 4,800句、dev 3,174句和test 3,569句；test由`house` 1,283句与`tech` 2,286句组成，其中974句含标注，共2,265个gold span。官方论文描述的14.5K句、12.5K以上span包含更完整的研究语料范围，不能直接当作本项目本地释放集规模。

## 任务与指标

数据提供两套BIO序列：`tags_skill`表示能力、行为与软技能，`tags_knowledge`表示技术、领域、方法和知识。评测器从BIO标签还原左闭右开的token span，并报告：

- `typed_exact`：起止边界与skill/knowledge类型全部一致；这是主要指标。
- `boundary_exact`：边界一致即可，忽略类型，用于分离分类错误与边界错误。
- `boundary_overlap`：预测与gold有token交叠即匹配，每个gold最多匹配一次；只作为宽松诊断，不能代替exact。
- 每类typed exact Precision、Recall与F1。

## 可复现运行

```powershell
.\scripts\download_skillspan.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements-skillspan.txt
.\.venv\Scripts\python.exe scripts\evaluate_skillspan.py --split test --mode catalog
.\.venv\Scripts\python.exe scripts\evaluate_skillspan.py --split test --mode jobbert
```

数据和模型均下载到D盘项目下的`data/external/`，报告写入被Git排除的`data/derived/`。JobBERT代码固定两个模型revision并要求Safetensors，避免浮动版本改变结果。

## 完整test结果

| 方法 | 评测范围 | Typed exact P/R/F1 | Boundary exact F1 | Overlap F1 | 失败 |
|---|---:|---:|---:|---:|---:|
| 技能词表规则 | 3,569句 / 2,265 spans | 0.6766 / 0.0702 / 0.1272 | 0.1272 | 0.1576 | 0 |
| 双JobBERT固定版本 | 3,569句 / 2,265 spans | 0.5593 / 0.6287 / 0.5920 | 0.6090 | 0.8032 | 0 |

JobBERT完整集上skill typed exact F1为0.5186，knowledge为0.6539。当前Windows CPU环境总耗时204.594秒，其中模型初始化11.017秒、批量推理193.575秒、评分0.001秒；该时间只用于本机复现记录，不代表服务吞吐。

## 固定80句诊断对照

固定种子`20260901`分层抽取80句（house/tech各40句，60个正例句、159个gold span），用于比较提示式抽取与专用模型，而不替代完整test结果：

| 方法 | Typed exact F1 | Boundary exact F1 | Overlap F1 | 说明 |
|---|---:|---:|---:|---|
| 技能词表规则 | 0.1071 | 0.1071 | 0.1071 | Precision 1.0但Recall仅0.0566 |
| Qwen3-4B零/少样本JSON抽取 | 0.1453 | 0.1844 | 0.5754 | 11个无效span被拒绝，0次调用失败 |
| 双JobBERT固定版本 | 0.5852 | 0.5981 | 0.8167 | 0个无效span，0次推理失败 |

Qwen3-4B经常找到大致区域，却会拆碎或扩张短语边界，因此overlap明显高于exact。JobBERT是使用SkillSpan任务监督训练的同域专用模型，而Qwen3-4B是通用生成模型提示抽取；该对照用于做工程选型，不是公平的模型架构排行榜，也不能写成“TalentMatch准确率提升”。

## 使用结论

当前结果支持把通用LLM保留在JD结构化和复杂语义复核环节，但英文岗位技能Span若进入生产链路，应优先采用专用token-classification endpoint或重新训练的抽取器，并继续做跨域、中文和独立人工集验证。JobBERT目前只接入离线benchmark，没有替换线上TalentMatch Analyzer。
