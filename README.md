# TalentMatch Agent

证据驱动的多智能体人岗匹配与招聘决策支持系统。系统接收一个岗位描述和一批候选人材料，完成岗位要求结构化、简历证据抽取、透明匹配、冲突复核、合规审计与候选人排序。

> 项目定位是招聘决策支持，不替代招聘人员作出录用或淘汰决定。

## 当前可运行能力

- 解析 TXT、PDF、DOCX 候选人材料；扫描版 PDF 会明确提示需要 OCR。
- 将岗位要求区分为硬性条件、加分项和上下文职责，并保留原文引用。
- 排序前由招聘人员检查、修改优先级、删除误抽取项并确认岗位条件。
- 从工作、项目、技能列表和自我评价中抽取不同强度的能力证据。
- 使用确定性权重完成评分；姓名、性别、年龄等敏感属性不参与计算。
- 对硬性条件缺失、技能只有弱证据、时间线不完整等情况进入复核分支。
- 使用 LangGraph 保存清晰的节点、条件路由和执行轨迹。
- 支持 `rules` 快速模式、按证据质量选择性调用模型的 `adaptive` 模式，以及完整 Ollama `qwen2.5:7b` + `embeddinggemma` 模式。
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
- 提供60份中文受控对抗样本，以及JTH公开招聘历史的规模化排序适配器；两类结果分开报告。

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
.\.venv\Scripts\python.exe -m pytest backend\tests
.\.venv\Scripts\python.exe scripts\evaluate_demo.py
.\.venv\Scripts\python.exe scripts\compare_baselines.py
.\.venv\Scripts\python.exe scripts\reliability_audit.py
.\.venv\Scripts\python.exe scripts\ollama_smoke.py
.\.venv\Scripts\python.exe scripts\evaluate_controlled.py
.\scripts\download_jth.ps1
.\.venv\Scripts\python.exe scripts\evaluate_jth.py
.\.venv\Scripts\python.exe scripts\evaluate_agent_ablation.py
.\.venv\Scripts\python.exe scripts\security_audit.py
.\.venv\Scripts\python.exe scripts\benchmark_parallel_ollama.py --mode adaptive
.\.venv\Scripts\python.exe scripts\compare_performance.py
```

演示结果写入 `data/*.json`，扩展评测写入被版本控制排除的 `data/derived/`。JTH原始数据同样不会提交或二次分发。

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
TALENTMATCH_CHAT_MODEL=qwen2.5:7b
TALENTMATCH_EMBED_MODEL=embeddinggemma
TALENTMATCH_OLLAMA_WORKERS=2
```

如果 Ollama 不可用，选择前端的“快速规则模式”，整个核心匹配和审计流程仍可运行。Ollama 模式中，所有模型抽取结果仍会进行原文引用校验；引用无法在输入中找到时会被丢弃或回退到规则分析。

模型JSON缓存保存在本地 `data/talentmatch.db`，不会上传外部服务。候选人抽取和模糊案例复核默认使用2路受控并行；存在明确严重冲突时只保留确定性结论，不重复调用LLM。完整Ollama模式固定8人无缓存耗时182.286秒；自适应模式先执行规则质量检查，只将3/8份弱证据材料送入LLM抽取，耗时降至76.744秒。相同输入缓存复跑最低观测为0.241秒。SQLite中仍可能保存结构化简历证据，因此企业化部署前需要增加加密、租户隔离和数据保留策略。

## 评测说明

项目把评测拆成三层：固定演示集验证端到端链路，中文受控合成集验证字段、否定句、弱证据和敏感属性边界，JTH公开招聘历史验证规模化排序。三者都不等于真实企业招聘效果，不能混写。当前已报告：

- 中文受控集：6类岗位、60份候选人材料，字段抽取、冲突复核消融、排序与成对敏感属性检查；
- JTH弱标签集：366个测试岗位、8,302个人岗对，比较关键词与结构化匹配；
- 固定演示集与本地Ollama真实链路：验证前后端、证据引用、持久化和回退路径。
- 固定8人Agent消融：单LLM直评、LLM证据抽取、确定性评分和冲突复核使用真实本地模型调用；
- 6类提示注入审计：验证攻击文本在进入LLM前被拦截，且确定性评分不受攻击指令影响。

详细规范见 [docs/evaluation_protocol.md](docs/evaluation_protocol.md)，JTH口径见 [docs/jth_benchmark.md](docs/jth_benchmark.md)。

## 明确边界

- 不抓取登录后的招聘平台或 LinkedIn 数据。
- 不使用姓名、性别、年龄、婚育状态、照片等信息评分。
- 不把历史录用结果直接视为无偏真值。
- 不自动拒绝、录用或联系候选人。
- 不允许匿名访问简历、岗位分析、运行记录与人工决策；管理员账号必须通过部署主机本地命令初始化。
- “硬性条件证据不足”只表示材料中没有足够证据，不代表候选人实际不具备该能力。
- 不把演示集结果描述为生产效果。
- 不承诺扫描件 OCR、企业 ATS 集成或实时招聘网站爬取已经完成。
- 默认开发配置仍使用进程内任务队列；完整容器配置使用Redis + RQ并经API重启恢复验证。当前未验证多主机调度、滚动升级和大规模并发容量。

## 技术栈

- FastAPI、Pydantic、LangGraph、SQLAlchemy、Alembic、SQLite/PostgreSQL
- Redis、RQ、Docker Compose、Nginx
- Ollama、Qwen2.5-7B、EmbeddingGemma
- PyMuPDF/PyPDF、python-docx
- React、Vite
- Pytest、自定义排序评测
- Prometheus Client、GitHub Actions
