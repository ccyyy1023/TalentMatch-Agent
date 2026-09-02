# TalentMatch Agent

证据驱动的多智能体人岗匹配与招聘决策支持系统。系统接收一个岗位描述和一批候选人材料，完成岗位要求结构化、简历证据抽取、透明匹配、冲突复核、合规审计与候选人排序。

> 项目定位是招聘决策支持，不替代招聘人员作出录用或淘汰决定。

> 隐私说明：仓库内岗位、候选人姓名、联系方式和履历均为虚构演示数据，不包含真实候选人简历。运行时上传文件、数据库、模型缓存以及SkillSpan、JTH、TalentCLEF原始数据均由`.gitignore`排除，不会随源码发布。

## 当前可运行能力

- 解析 TXT、PDF、DOCX 候选人材料；扫描版 PDF 会明确提示需要 OCR。
- 将岗位要求区分为硬性条件、加分项和上下文职责，并保留原文引用。
- 排序前由招聘人员检查、修改优先级、删除误抽取项并确认岗位条件。
- 从工作、项目、技能列表和自我评价中抽取不同强度的能力证据。
- 使用硬性条件85%、加分项10%、上下文5%的分组权重完成评分；组内先求平均，避免冗长JD中的上下文条目靠数量稀释硬性要求；姓名、性别、年龄等敏感属性不参与计算。
- 对硬性条件缺失、技能只有弱证据、时间线不完整等情况进入复核分支。
- 使用 LangGraph 保存清晰的节点、条件路由和执行轨迹。
- 支持 `rules` 快速模式、按证据质量选择性调用模型的 `adaptive` 模式，以及完整 Ollama `qwen3:4b` + `embeddinggemma` 模式。
- 使用“文本SHA-256语义等价键 + 模型摘要 + 提示词版本 + 参数”持久化缓存模型JSON结果。
- 检测简历中的提示注入指令；命中时跳过LLM解析，强制走确定性证据路径并给出安全告警。
- 支持SQLite本地开发和PostgreSQL部署；通过Alembic管理运行记录、评测、账号、岗位、候选人和审计表结构。
- 采用管理员预创建账号、管理员/招聘人员两级角色、8小时持久会话和操作审计保护候选人材料；不开放公共注册。
- 长耗时分析可使用Redis + RQ持久化队列，API重启后任务状态仍可恢复，由独立Worker执行并写入运行记录。
- 提供 React 招聘工作台以及岗位管理、候选人库、运行记录和管理员审计页面。
- 提供持久化筛选批次：从岗位库和候选人库直接发起异步分析，关联任务、运行记录、逐人决策与候选人筛选历史。
- 输出结构化JSON请求日志与Prometheus指标，覆盖HTTP请求、任务提交/结果、队列深度、排队等待、端到端耗时和Agent节点耗时。
- 提供GitHub Actions工作流，分别执行后端完整测试和前端生产构建。
- 内置带三级相关性标签的演示集和 NDCG@5、Precision@3、Recall@5、MRR 等排序指标。
- 提供中文受控集、SkillSpan、TalentCLEF与JTH四层评测；规则边界、真实岗位技能Span、完整文本排序和规模化结构化排序分开报告。
- 提供可开关的英文JobBERT岗位技能补充与岗位条件驱动候选人核验；该实验路径默认关闭，只有通过目标数据验证后才允许启用。

## 架构

```text
JD + Candidate Documents
        |
        v
JD Analyzer Agent -------- 保留岗位原文证据
        |
        v
Human Criteria Review ---- 修改优先级/删除误抽取项/确认条件
        |
        v
Candidate Evidence Agents - 区分项目/工作/技能列表证据强度
        |
        v
Deterministic Matcher ----- 硬性条件 + 权重 + 证据强度
        |
        +---- 无异常 -----> Compliance Agent
        |
        +---- 弱证据/冲突 -> Conflict Reviewer Agent
                              |
                              v
                     Compliance Agent
                              |
                              v
                 Ranking + Evidence Explanation
```

大模型不直接生成最终分数。它负责结构化抽取与复核，最终分数由可检查的规则计算。

## 快速启动

### 1. 安装

在项目根目录运行：

```powershell
.\scripts\setup.ps1
```

首次部署后创建管理员（密码采用隐藏输入，不会出现在命令历史中）：

```powershell
.\.venv\Scripts\python.exe scripts\create_user.py --username admin --role admin
```

管理员登录后可通过`POST /api/v1/users`创建招聘人员账号。系统不提供公开注册。

### 2. 启动后端

```powershell
.\scripts\start_backend.ps1
```

接口文档：<http://127.0.0.1:8000/docs>

长任务接口：`POST /api/v1/analysis/tasks` 创建任务，`GET /api/v1/analysis/tasks/{task_id}` 查询阶段、进度与最终结果。原同步分析接口继续保留用于兼容和自动化测试。

企业筛选闭环：先通过`POST /api/v1/jobs`和`POST /api/v1/candidates`保存业务对象，再调用`POST /api/v1/screening-batches`选择岗位与候选人并发起批次。批次状态由`GET /api/v1/screening-batches/{batch_id}`持续同步；分析完成后通过候选人决策接口记录“进入下一轮/保持观察/暂不推进”，并可从`GET /api/v1/candidates/{candidate_id}/screening-history`追溯历史。

### 3. 启动前端

另开一个终端：

```powershell
.\scripts\start_frontend.ps1
```

浏览器访问：<http://127.0.0.1:5173>

### 4. 运行测试与评测

```powershell
Push-Location backend
& ..\.venv\Scripts\python.exe -m pytest tests
Pop-Location
.\.venv\Scripts\python.exe scripts\evaluate_demo.py
.\.venv\Scripts\python.exe scripts\compare_baselines.py
.\.venv\Scripts\python.exe scripts\reliability_audit.py
.\.venv\Scripts\python.exe scripts\ollama_smoke.py
.\.venv\Scripts\python.exe scripts\evaluate_controlled.py
.\scripts\download_skillspan.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements-skillspan.txt
.\.venv\Scripts\python.exe scripts\evaluate_skillspan.py --split test --mode catalog
.\.venv\Scripts\python.exe scripts\evaluate_skillspan.py --split test --mode jobbert
.\scripts\download_jth.ps1
.\.venv\Scripts\python.exe scripts\evaluate_jth.py
.\scripts\download_talentclef.ps1
.\.venv\Scripts\python.exe scripts\evaluate_talentclef.py --language en
.\.venv\Scripts\python.exe scripts\evaluate_talentclef.py --language es
.\.venv\Scripts\python.exe scripts\evaluate_talentclef_extraction_ab.py --left-model qwen2.5:7b --right-model qwen3:4b
.\.venv\Scripts\python.exe scripts\evaluate_talentclef_hybrid.py
.\.venv\Scripts\python.exe scripts\evaluate_talentclef_hard_ab.py --query-limit 10
.\.venv\Scripts\python.exe scripts\evaluate_talentclef_agent_ablation.py
.\.venv\Scripts\python.exe scripts\evaluate_reviewer_heterogeneity.py
.\.venv\Scripts\python.exe scripts\evaluate_agent_ablation.py
.\.venv\Scripts\python.exe scripts\security_audit.py
.\.venv\Scripts\python.exe scripts\benchmark_parallel_ollama.py --mode adaptive
.\.venv\Scripts\python.exe scripts\benchmark_agent_models.py --mode adaptive --jd-model qwen3:4b --candidate-model qwen3:4b --reviewer-model qwen2.5:7b
.\.venv\Scripts\python.exe scripts\compare_performance.py
```

演示结果写入 `data/*.json`，扩展评测写入被版本控制排除的 `data/derived/`。SkillSpan、JTH与TalentCLEF原始数据和SkillSpan模型缓存同样不会提交或二次分发。

### 5. 完整容器化部署

完整部署包含Nginx前端、FastAPI、独立RQ Worker、PostgreSQL、Redis和自动Alembic迁移。先在当前终端设置两个不同的强密码，再启动：

```powershell
$env:TALENTMATCH_POSTGRES_PASSWORD='替换为强密码'
$env:TALENTMATCH_REDIS_PASSWORD='替换为另一个强密码'
docker compose -f deploy\compose.production.yml up -d --build
```

首次启动后创建内部管理员：

```powershell
$env:TALENTMATCH_INITIAL_PASSWORD='替换为管理员初始密码'
docker compose -f deploy\compose.production.yml exec -e TALENTMATCH_INITIAL_PASSWORD api python /app/scripts/create_user.py --username admin --role admin
```

默认入口为 <http://127.0.0.1:8080>。数据库和Redis不向宿主机公开端口；前端通过Nginx同源代理访问API。

需要增加规则任务消费能力时，可横向增加Worker实例：

```powershell
docker compose -f deploy\compose.production.yml up -d --scale worker=2 worker
```

RQ任务状态保存在Redis中，API重启不会删除已排队任务；Nginx通过Docker内置DNS重新解析重建后的API容器地址。
Worker名称包含容器hostname，使用`--scale worker=N`时不会因所有容器PID均为1而发生RQ名称冲突。

Prometheus可在容器网络内抓取API的`/metrics`，当前只提供可解释的核心服务指标，不预设Kubernetes、OpenTelemetry或复杂告警平台。结构化日志写入标准输出，由部署环境决定是否接入日志平台。

## Ollama

默认配置：

```env
TALENTMATCH_OLLAMA_BASE_URL=http://127.0.0.1:11434
TALENTMATCH_CHAT_MODEL=qwen3:4b
TALENTMATCH_JD_MODEL=qwen3:4b
TALENTMATCH_CANDIDATE_MODEL=qwen3:4b
TALENTMATCH_REVIEWER_MODEL=qwen2.5:7b
TALENTMATCH_EMBED_MODEL=embeddinggemma
TALENTMATCH_SKILL_EXTRACTOR=catalog
TALENTMATCH_JOBBERT_CACHE_DIR=data/external/skillspan_models
TALENTMATCH_OLLAMA_WORKERS=2
```

`TALENTMATCH_CHAT_MODEL`保留为统一兼容配置，三个Agent变量可以单独覆盖。当前默认让JD与候选人抽取使用Qwen3-4B，让Conflict Reviewer使用Qwen2.5-7B；Reviewer输出还必须经过确定性证据质量门。API返回的`model_info.agent_models`和健康检查会公开每个Agent实际使用的模型，避免配置与运行状态不一致。

`TALENTMATCH_SKILL_EXTRACTOR=jobbert`会为英文JD补充固定版本JobBERT开放技能Span，并在候选人原文中只核验当前岗位所需技能。该路径不会对中文或西班牙文调用英文模型，模型懒加载且失败时保留原有规则/LLM结果。TalentCLEF完整development和难负样本实验未证明其跨岗位稳定改善，因此默认保持`catalog`，不得写成线上效果提升。

如果 Ollama 不可用，选择前端的“快速规则模式”，整个核心匹配和审计流程仍可运行。Ollama 模式中，所有模型抽取结果仍会进行原文引用校验；引用无法在输入中找到时会被丢弃或回退到规则分析。

模型JSON缓存保存在本地 `data/talentmatch.db`，不会上传外部服务。候选人抽取和模糊案例复核默认使用2路受控并行；存在明确严重冲突时只保留确定性结论，不重复调用LLM。历史Qwen2.5-7B固定8人测试中，完整Ollama模式无缓存耗时182.286秒；自适应模式先执行规则质量检查，只将3/8份弱证据材料送入LLM抽取，耗时降至76.744秒，相同输入缓存复跑最低观测为0.241秒。TalentCLEF固定3个JD/17份CV的冷缓存抽取诊断中，Qwen3-4B为321.455秒且零回退，Qwen2.5-7B为555.795秒且2次回退，两者样本排序均为MAP 1.0；该开发集诊断不等于封存测试或生产效果。Qwen3-8B在当前8GB显存设备的全文预检中持续触发120秒超时，因此不作为默认抽取模型。SQLite中仍可能保存结构化简历证据，因此企业化部署前需要增加加密、租户隔离和数据保留策略。

## 评测说明

项目把数据评测拆成四层：中文受控合成集验证规则边界，SkillSpan验证真实英文岗位句子的技能Span抽取，TalentCLEF完整JD/CV文本验证端到端候选人排序，JTH公开招聘历史验证规模化结构化排序。固定演示集另用于工程链路回归。各层任务、标签和指标不同，都不等于真实企业招聘效果，不能混写。当前已报告：

- 中文受控集：6类岗位、60份候选人材料，覆盖48项JD内容类别与优先级联合标注，以及字段抽取、冲突复核消融、排序与成对敏感属性检查；
- SkillSpan：官方test释放集3,569句、2,265个gold span；固定版本双JobBERT端点的typed exact F1为0.5920、boundary exact F1为0.6090、boundary overlap F1为0.8032，0次推理失败；该结果是同域监督模型专项基线，不是TalentMatch端到端效果；
- JTH弱标签集：使用2022年岗位调参、2023年岗位选择配置、2024年后366个岗位与8,302个人岗对封存测试；同时比较技能关键词、原始字段多属性基线与结构化匹配，结构化匹配NDCG@5为0.5239；
- TalentCLEF 2026 Task A：接入英/西双语完整职位与简历文本，development每种语言包含10个JD、472份CV和472条专家二元相关标注；BM25用于验证原始文本、全库排序、官方指标及TREC输出链路；固定3个JD/17份CV的Qwen抽取诊断中，Qwen3-4B在保持相同样本排序的同时耗时更低、回退更少，但该开发切片不作为封存效果；
- TalentCLEF优化复核：完整英文development上的JobBERT技能融合只取得极小内部留出增量；10个JD、60个人岗难例对上，Qwen证据匹配NDCG为0.7761，而加入JobBERT后降至0.6557；两阶段融合也未通过3-JD留出验证，因此相关路径保留为实验开关而非默认能力；
- 单Agent对照：在5个JD、30个人岗难例对上，单Qwen直接评分NDCG为0.7424，高于同范围证据工作流0.6479，但其双侧原文引用有效率为96.67%，低于证据工作流100%；该小型development消融只用于说明排序与可靠性的取舍，不证明Multi-Agent普遍更优；
- 固定演示集与本地Ollama真实链路：验证前后端、证据引用、持久化和回退路径。
- 固定8人Agent消融：单LLM直评、LLM证据抽取、确定性评分和冲突复核使用真实本地模型调用；
- Reviewer异构模型消融：先观察到`qwen2.5:7b` Reviewer查全率更高但产生5条误报；加入确定性证据质量门后，同一24条受控边界样例上误报降为0，配对置信区间通过预设启用门槛，因此将Reviewer独立配置为Qwen2.5-7B；该结果仍不是独立人工集或生产效果；
- 6类提示注入审计：验证攻击文本在进入LLM前被拦截，且确定性评分不受攻击指令影响。

详细规范见 [docs/evaluation_protocol.md](docs/evaluation_protocol.md)，SkillSpan口径见 [docs/skillspan_benchmark.md](docs/skillspan_benchmark.md)，JTH口径见 [docs/jth_benchmark.md](docs/jth_benchmark.md)，TalentCLEF口径见 [docs/talentclef_benchmark.md](docs/talentclef_benchmark.md)，Reviewer消融见 [docs/reviewer_heterogeneity.md](docs/reviewer_heterogeneity.md)。

## 明确边界

- 不抓取登录后的招聘平台或 LinkedIn 数据。
- 不使用姓名、性别、年龄、婚育状态、照片等信息评分。
- 不把历史录用结果直接视为无偏真值。
- 不自动拒绝、录用或联系候选人。
- 不允许匿名访问简历、岗位分析、运行记录与人工决策；管理员账号必须通过部署主机本地命令初始化。
- “硬性条件证据不足”只表示材料中没有足够证据，不代表候选人实际不具备该能力。
- 不把演示集结果描述为生产效果。

## License

项目自有代码采用 [MIT License](LICENSE) 发布。第三方数据集和模型仍分别遵循其原始许可与使用条件，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- 不承诺扫描件 OCR、企业 ATS 集成或实时招聘网站爬取已经完成。
- 默认开发配置仍使用进程内任务队列；完整容器配置使用Redis + RQ并经API重启恢复验证。当前未验证多主机调度、滚动升级和大规模并发容量。

## 技术栈

- FastAPI、Pydantic、LangGraph、SQLAlchemy、Alembic、SQLite/PostgreSQL
- Redis、RQ、Docker Compose、Nginx
- Ollama、Qwen3-4B、EmbeddingGemma
- PyMuPDF/PyPDF、python-docx
- React、Vite
- Pytest、自定义排序评测
- Prometheus Client、GitHub Actions
