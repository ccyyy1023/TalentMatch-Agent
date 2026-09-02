# TalentCLEF 2026 Task A 接入与评测口径

## 数据来源

- 数据集：TalentCLEF 2026 Task A（Contextualized Job-Person Matching）v0.3.0
- 记录页：https://zenodo.org/records/19652670
- 概念 DOI：https://doi.org/10.5281/zenodo.17625261
- 许可证：CC BY 4.0
- 下载文件：`TaskA.zip`
- 官方 MD5：`431ec3b693ae1ba24fe04793f9c1f750`

Task A 提供英文和西班牙文的完整职位描述与简历文本。官方说明这些文本由真实招聘结构化数据合成，并经过人工审阅，以保留招聘语境同时保护隐私。文件中仍包含合成的姓名、联系方式等版式字段，因此项目将整个原始目录排除在版本控制之外，不将其视为可公开的真实候选人数据。

## 本地范围

| 划分 | 语言 | JD 查询 | CV 语料 | qrels | 本地用途 |
|---|---:|---:|---:|---:|---|
| development | en | 10 | 472 | 472 | 可计算公开指标 |
| development | es | 10 | 472 | 472 | 可计算公开指标 |
| test | en | 40 | 476 | 不公开 | 只生成 run，不报告本地效果 |
| test | es | 40 | 476 | 不公开 | 只生成 run，不报告本地效果 |

开发集 qrels 是专家给出的二元相关性标注。没有在某个 JD 下出现的 CV 按官方信息检索评测习惯视为未相关。由于每种语言仅有 10 个 JD，任何聚合结果都必须同时披露查询数，不能泛化为生产招聘效果。

## 可复现流程

```powershell
.\scripts\download_talentclef.ps1
.\.venv\Scripts\python.exe scripts\evaluate_talentclef.py --language en
.\.venv\Scripts\python.exe scripts\evaluate_talentclef.py --language es
```

下载脚本固定 Zenodo 版本并校验 MD5。评测脚本读取无扩展名 UTF-8 文本，校验 query、corpus 与 qrels ID 完整性，然后生成：

- `data/derived/talentclef_development_<lang>_bm25.json`：指标和逐查询结果；
- `data/derived/talentclef_development_<lang>_bm25.run`：官方评测代码可直接读取的无表头 TREC run 格式排名。

这两个目录都被 `.gitignore` 排除，不二次分发原始文本或派生排名。

## 当前基线和后续 A/B 的边界

第一阶段只运行无训练、无 LLM 调用的全文 BM25，目标是验证“原始文本读取—全库排序—官方指标—TREC 输出”链路。指标与官方 Task A 对齐：MAP、MRR、NDCG、Precision@5、Precision@10、Precision@100。

Qwen2.5 与 Qwen3 的抽取 A/B 必须在同一固定数据切片、同一提示模板、同一确定性下游评分和冷缓存条件下运行。当前硬件上的候选组是 Qwen2.5-7B 与 Qwen3-4B，属于工程部署选型而非同参数规模学术比较；此前 Qwen3-8B 在 8 GB 显存环境的全文预检中持续触发 120 秒超时，因此不扩大无效样本。A/B 结果需同时报告解析成功率、原文证据有效率、回退率、耗时与下游排序指标；不能仅凭少量示例或主观观察宣布模型升级有效。

默认 A/B 使用固定随机种子，从 3 个 JD 各取 4 个相关、4 个未相关 CV。每个模型只解析去重后的 CV，关闭持久化缓存，并把抽取后的要求与证据重新组成压缩文本，通过同一个 BM25 排序器评估信息保留能力。脚本在每个模型完成后落盘一次，长时间本地评测中断时仍保留已完成模型的证据。该切片使用 development qrels 做模型选择，因此只能称为诊断结果，不能称为封存测试成绩。

## 2026-09-01 初版 A/B 结果

为控制首轮本地耗时，实际运行参数为3个JD、每个JD 3个相关和3个未相关CV，去重后17份CV；每个模型共20次冷缓存调用。Qwen2.5-7B耗时555.795秒，发生1次JD回退和1次CV回退；Qwen3-4B耗时321.455秒，无回退，耗时降低42.2%。两种抽取结果在该切片上的MAP、MRR、NDCG均为1.0，未观察到排序质量差异。

因此当前证据只支持“在该硬件和开发集切片上，Qwen3-4B以相同样本排序取得更低时延和更少回退”的工程结论，不支持“Qwen3纯模型质量更高”或“真实招聘效果提升”。Qwen3-8B在同机全文和2500字符探针中均触发120秒超时，已从默认抽取候选中排除。派生报告位于`data/derived/talentclef_extraction_ab.json`且不纳入版本控制。
