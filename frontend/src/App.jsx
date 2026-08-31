import { useEffect, useMemo, useState } from 'react'
import {
  Activity, AlertTriangle, ArrowRight, Bot, BriefcaseBusiness, CheckCircle2,
  ChevronDown, ChevronUp, CircleUserRound, Database, FileCheck2, Gauge,
  ListChecks, LoaderCircle, LockKeyhole, LogIn, LogOut, Play, RefreshCw, SearchCheck, ShieldCheck, Sparkles, UploadCloud, UsersRound, X,
  Trash2,
} from 'lucide-react'

const API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api/v1'

const recommendationMeta = {
  recommended: { label: '建议进入人工复核', className: 'success' },
  manual_review: { label: '需要人工复核', className: 'warning' },
  insufficient_hard_requirement_evidence: { label: '硬性条件证据不足', className: 'danger' },
  hard_requirement_not_met: { label: '硬性条件证据不足', className: 'danger' },
}

const statusMeta = {
  matched: ['已匹配', 'success'], partial: ['部分匹配', 'warning'],
  missing: ['缺少证据', 'danger'], review: ['待复核', 'neutral'],
}

const taskStageLabels = {
  queued: '等待执行', starting: '初始化工作流', job_ready: '岗位条件就绪',
  candidates_ready: '候选人证据抽取完成', matching_ready: '透明匹配完成',
  review_ready: '冲突复核完成', compliance_ready: '合规检查完成',
  ranking_ready: '候选人排序完成', completed: '分析完成', failed: '分析失败',
}

const wait = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds))

function App() {
  const [authToken, setAuthToken] = useState(() => localStorage.getItem('talentmatch_token') || '')
  const [authUser, setAuthUser] = useState(null)
  const [authChecking, setAuthChecking] = useState(true)
  const [loginUsername, setLoginUsername] = useState('')
  const [loginPassword, setLoginPassword] = useState('')
  const [demo, setDemo] = useState(null)
  const [jobDescription, setJobDescription] = useState('')
  const [candidates, setCandidates] = useState([])
  const [mode, setMode] = useState('rules')
  const [result, setResult] = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const [expandedTrace, setExpandedTrace] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [health, setHealth] = useState(null)
  const [decisionStatus, setDecisionStatus] = useState('')
  const [parsedJob, setParsedJob] = useState(null)
  const [criteriaConfirmed, setCriteriaConfirmed] = useState(false)
  const [parsingJob, setParsingJob] = useState(false)
  const [taskProgress, setTaskProgress] = useState(0)
  const [taskStage, setTaskStage] = useState('')
  const [activeView, setActiveView] = useState('workbench')
  const [managedJobs, setManagedJobs] = useState([])
  const [managedCandidates, setManagedCandidates] = useState([])
  const [runHistory, setRunHistory] = useState([])
  const [auditRecords, setAuditRecords] = useState([])
  const [screeningBatches, setScreeningBatches] = useState([])

  useEffect(() => {
    fetch(`${API}/health`).then(r => r.json()).then(setHealth).catch(() => setHealth(null))
  }, [])

  useEffect(() => {
    if (!authToken) {
      setAuthChecking(false)
      return
    }
    fetch(`${API}/auth/me`, { headers: { Authorization: `Bearer ${authToken}` } })
      .then(async response => {
        if (!response.ok) throw new Error('登录已失效')
        setAuthUser(await response.json())
      })
      .catch(() => {
        localStorage.removeItem('talentmatch_token')
        setAuthToken('')
        setAuthUser(null)
      })
      .finally(() => setAuthChecking(false))
  }, [authToken])

  useEffect(() => {
    if (!authUser || !authToken) return
    apiFetch(`${API}/demo`).then(async response => {
      if (!response.ok) throw new Error('无法加载演示数据')
      const data = await response.json()
      setDemo(data)
      setJobDescription(data.job_description)
      setCandidates(data.candidates)
    }).catch(err => setError(`后端连接失败：${err.message}`))
  }, [authUser, authToken])

  useEffect(() => {
    if (!authUser || activeView === 'workbench') return
    if (activeView === 'batches') {
      Promise.all([
        apiFetch(`${API}/jobs`).then(response => response.json()),
        apiFetch(`${API}/candidates`).then(response => response.json()),
        apiFetch(`${API}/screening-batches`).then(response => response.json()),
      ]).then(([jobs, candidates, batches]) => {
        setManagedJobs(jobs)
        setManagedCandidates(candidates)
        setScreeningBatches(batches)
      }).catch(err => setError(err.message))
      return
    }
    const endpoint = { jobs: 'jobs', candidates: 'candidates', runs: 'runs', audit: 'audit' }[activeView]
    apiFetch(`${API}/${endpoint}`).then(async response => {
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || '数据加载失败')
      if (activeView === 'jobs') setManagedJobs(payload)
      if (activeView === 'candidates') setManagedCandidates(payload)
      if (activeView === 'runs') setRunHistory(payload)
      if (activeView === 'audit') setAuditRecords(payload)
    }).catch(err => setError(err.message))
  }, [activeView, authUser, authToken])

  function apiFetch(url, options = {}) {
    return fetch(url, {
      ...options,
      headers: { ...(options.headers || {}), Authorization: `Bearer ${authToken}` },
    })
  }

  async function handleLogin(event) {
    event.preventDefault()
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${API}/auth/login`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: loginUsername, password: loginPassword }),
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || '登录失败')
      localStorage.setItem('talentmatch_token', payload.access_token)
      setAuthToken(payload.access_token)
      setAuthUser(payload.user)
      setLoginPassword('')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleLogout() {
    try {
      await apiFetch(`${API}/auth/logout`, { method: 'POST' })
    } finally {
      localStorage.removeItem('talentmatch_token')
      setAuthToken('')
      setAuthUser(null)
      setDemo(null)
      setResult(null)
    }
  }

  const selected = useMemo(
    () => result?.ranking.find(item => item.candidate_id === selectedId) || result?.ranking[0],
    [result, selectedId],
  )

  async function runAnalysis() {
    if (!demo || loading || !parsedJob || !criteriaConfirmed) return
    setLoading(true)
    setError('')
    setTaskProgress(0)
    setTaskStage('queued')
    try {
      const response = await apiFetch(`${API}/analysis/tasks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...demo, job_description: jobDescription, candidates, mode,
          confirmed_job: parsedJob, criteria_confirmed_by_human: true,
        }),
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || '分析失败')
      let completed = null
      while (!completed) {
        const statusResponse = await apiFetch(`${API}/analysis/tasks/${payload.task_id}`)
        const status = await statusResponse.json()
        if (!statusResponse.ok) throw new Error(status.detail || '无法读取分析任务状态')
        setTaskProgress(status.progress)
        setTaskStage(status.stage)
        if (status.status === 'failed') throw new Error(status.error || '分析任务失败')
        if (status.status === 'completed') {
          completed = status.result
          break
        }
        await wait(700)
      }
      setResult(completed)
      setSelectedId(completed.ranking[0]?.candidate_id)
      setDecisionStatus('')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function parseJobRequirements() {
    if (jobDescription.trim().length < 20 || parsingJob) return
    setParsingJob(true)
    setError('')
    try {
      const response = await apiFetch(`${API}/jobs/parse`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: jobDescription, mode }),
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || '岗位条件解析失败')
      setParsedJob(payload.job)
      setCriteriaConfirmed(false)
    } catch (err) {
      setError(err.message)
    } finally {
      setParsingJob(false)
    }
  }

  function updateRequirement(id, priority) {
    setParsedJob(current => ({
      ...current,
      requirements: current.requirements.map(item => item.id === id ? { ...item, priority } : item),
    }))
    setCriteriaConfirmed(false)
  }

  function removeRequirement(id) {
    setParsedJob(current => ({ ...current, requirements: current.requirements.filter(item => item.id !== id) }))
    setCriteriaConfirmed(false)
  }

  async function saveDecision(decision) {
    if (!result || !selected) return
    setDecisionStatus('保存中...')
    try {
      const response = await apiFetch(`${API}/runs/${result.run_id}/decisions`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidate_id: selected.candidate_id, decision, note: '招聘工作台人工复核记录' }),
      })
      if (!response.ok) throw new Error('保存失败')
      const labels = { advance: '已标记进入下一轮', hold: '已标记保持观察', not_advance: '已标记暂不推进' }
      setDecisionStatus(labels[decision])
    } catch (err) {
      setDecisionStatus(err.message)
    }
  }

  async function handleFiles(event) {
    const files = Array.from(event.target.files || [])
    if (!files.length) return
    setLoading(true)
    setError('')
    try {
      const parsed = []
      for (const file of files) {
        const form = new FormData()
        form.append('file', file)
        const response = await apiFetch(`${API}/documents/parse`, { method: 'POST', body: form })
        const payload = await response.json()
        if (!response.ok) throw new Error(`${file.name}：${payload.detail || '解析失败'}`)
        parsed.push({ id: `upload-${Date.now()}-${parsed.length + 1}`, name: file.name.replace(/\.[^.]+$/, ''), text: payload.text })
      }
      setCandidates(current => [...current, ...parsed].slice(0, 50))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      event.target.value = ''
    }
  }

  async function saveCurrentJob() {
    const response = await apiFetch(`${API}/jobs`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: parsedJob?.title || '待命名岗位', description: jobDescription, status: 'draft' }),
    })
    const payload = await response.json()
    if (!response.ok) throw new Error(payload.detail || '岗位保存失败')
    setManagedJobs(current => [payload, ...current])
    setActiveView('jobs')
  }

  async function saveCurrentCandidates() {
    const saved = []
    for (const candidate of candidates) {
      const response = await apiFetch(`${API}/candidates`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ display_name: candidate.name || candidate.id, external_ref: candidate.id, resume_text: candidate.text }),
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || '候选人保存失败')
      saved.push(payload)
    }
    setManagedCandidates(current => [...saved, ...current])
    setActiveView('candidates')
  }

  async function createScreeningBatch(jobId, candidateIds, batchMode) {
    setError('')
    const response = await apiFetch(`${API}/screening-batches`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId, candidate_ids: candidateIds, mode: batchMode }),
    })
    let batch = await response.json()
    if (!response.ok) throw new Error(batch.detail || '筛选批次创建失败')
    setScreeningBatches(current => [batch, ...current.filter(item => item.id !== batch.id)])
    while (['queued', 'running'].includes(batch.status)) {
      await wait(700)
      const statusResponse = await apiFetch(`${API}/screening-batches/${batch.id}`)
      batch = await statusResponse.json()
      if (!statusResponse.ok) throw new Error(batch.detail || '无法读取筛选批次状态')
      setScreeningBatches(current => [batch, ...current.filter(item => item.id !== batch.id)])
    }
    return batch
  }

  async function saveBatchDecision(batchId, candidateId, decision) {
    const response = await apiFetch(`${API}/screening-batches/${batchId}/candidates/${candidateId}/decision`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision, note: '筛选批次页面人工复核记录' }),
    })
    const batch = await response.json()
    if (!response.ok) throw new Error(batch.detail || '人工决策保存失败')
    setScreeningBatches(current => current.map(item => item.id === batch.id ? batch : item))
    return batch
  }

  if (authChecking) {
    return <div className="auth-loading"><LoaderCircle className="spin"/><span>正在验证企业会话</span></div>
  }

  if (!authUser) {
    return <div className="login-shell">
      <section className="login-card">
        <div className="login-brand"><div className="brand-mark"><Sparkles size={21}/></div><div><b>TalentMatch</b><span>Agent</span></div></div>
        <div className="login-icon"><LockKeyhole size={27}/></div>
        <p className="eyebrow">INTERNAL RECRUITMENT SYSTEM</p>
        <h1>企业内部登录</h1>
        <p className="login-description">系统包含候选人敏感材料，仅限管理员创建的内部账号访问。</p>
        <form onSubmit={handleLogin}>
          <label>账号<input value={loginUsername} onChange={event => setLoginUsername(event.target.value)} autoComplete="username" required minLength="3"/></label>
          <label>密码<input type="password" value={loginPassword} onChange={event => setLoginPassword(event.target.value)} autoComplete="current-password" required minLength="10"/></label>
          {error && <div className="login-error"><AlertTriangle size={15}/>{error}</div>}
          <button className="primary" disabled={loading}>{loading ? <LoaderCircle className="spin" size={17}/> : <LogIn size={17}/>}登录工作台</button>
        </form>
        <div className="login-security"><ShieldCheck size={14}/>无公开注册；会话与关键操作写入本地审计记录</div>
      </section>
    </div>
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark"><Sparkles size={21}/></div><div><b>TalentMatch</b><span>Agent</span></div></div>
        <nav>
          <button className={`nav-item ${activeView === 'workbench' ? 'active' : ''}`} onClick={() => setActiveView('workbench')}><Gauge size={18}/>匹配工作台</button>
          <button className={`nav-item ${activeView === 'jobs' ? 'active' : ''}`} onClick={() => setActiveView('jobs')}><BriefcaseBusiness size={18}/>岗位管理</button>
          <button className={`nav-item ${activeView === 'candidates' ? 'active' : ''}`} onClick={() => setActiveView('candidates')}><UsersRound size={18}/>候选人库</button>
          <button className={`nav-item ${activeView === 'batches' ? 'active' : ''}`} onClick={() => setActiveView('batches')}><ListChecks size={18}/>筛选批次</button>
          <button className={`nav-item ${activeView === 'runs' ? 'active' : ''}`} onClick={() => setActiveView('runs')}><Activity size={18}/>运行记录</button>
          {authUser.role === 'admin' && <button className={`nav-item ${activeView === 'audit' ? 'active' : ''}`} onClick={() => setActiveView('audit')}><ShieldCheck size={18}/>审计记录</button>}
        </nav>
        <div className="system-card">
          <div className="system-title"><span className={`status-dot ${health?.ollama?.available ? 'online' : ''}`}/><b>系统状态</b></div>
          <p>规则引擎：可用</p>
          <p>Ollama：{health?.ollama?.available ? '已连接' : '未连接'}</p>
          <p>模型缓存：{health?.model_cache?.entries ?? 0} 条</p>
          <p>当前账号：{authUser.username}（{authUser.role === 'admin' ? '管理员' : '招聘人员'}）</p>
          <p className="muted">敏感属性不参与评分</p>
          <button className="logout-button" onClick={handleLogout}><LogOut size={13}/>退出登录</button>
        </div>
      </aside>

      <main>
        {activeView === 'workbench' ? <>
        <header>
          <div><p className="eyebrow">RECRUITMENT DECISION SUPPORT</p><h1>证据驱动的人岗匹配</h1><p>系统提供排序建议和证据，不替代招聘人员作出决定。</p></div>
          <div className="header-badge"><ShieldCheck size={17}/>人工复核已启用</div>
        </header>

        <section className="input-grid">
          <div className="panel job-panel">
            <div className="panel-heading"><div><span className="step">01</span><h2>岗位描述</h2></div><span className="counter">{jobDescription.length} 字</span></div>
            <textarea value={jobDescription} onChange={e => { setJobDescription(e.target.value); setParsedJob(null); setCriteriaConfirmed(false) }} placeholder="粘贴岗位JD..."/>
            <div className="criteria-actions">
              <button onClick={parseJobRequirements} disabled={parsingJob || jobDescription.trim().length < 20}>
                {parsingJob ? <LoaderCircle className="spin" size={14}/> : <SearchCheck size={14}/>}
                {parsingJob ? '正在解析条件' : parsedJob ? '重新解析岗位条件' : '解析岗位条件'}
              </button>
              {parsedJob && <span className={criteriaConfirmed ? 'confirmed' : ''}>{criteriaConfirmed ? '已由招聘人员确认' : '请检查条件后确认'}</span>}
              <button onClick={() => saveCurrentJob().catch(err => setError(err.message))} disabled={jobDescription.trim().length < 20}><Database size={14}/>保存岗位</button>
            </div>
            {parsedJob && <div className="criteria-editor">
              {parsedJob.requirements.map(item => <div className="criteria-edit-row" key={item.id}>
                <span>{item.text}</span>
                <select value={item.priority} onChange={e => updateRequirement(item.id, e.target.value)}>
                  <option value="hard">硬性条件</option><option value="preferred">加分项</option><option value="context">上下文</option>
                </select>
                <button title="删除条件" onClick={() => removeRequirement(item.id)}><Trash2 size={13}/></button>
              </div>)}
              <button className="confirm-criteria" disabled={!parsedJob.requirements.length} onClick={() => setCriteriaConfirmed(true)}>
                <ShieldCheck size={14}/>{criteriaConfirmed ? '条件已确认' : `确认 ${parsedJob.requirements.length} 项条件`}
              </button>
            </div>}
          </div>
          <div className="panel candidate-panel">
            <div className="panel-heading"><div><span className="step">02</span><h2>候选人批次</h2></div><span className="counter">最多50份</span></div>
            <div className="candidate-summary">
              <div className="stack-icon"><FileCheck2 size={25}/></div>
              <div><b>{candidates.length} 份候选人档案</b><p>支持TXT、PDF和DOCX，展示时统一匿名化</p></div>
            </div>
            <div className="candidate-tags">
              {candidates.slice(0, 5).map(item => <span key={item.id}>{item.id}</span>)}
              {candidates.length > 5 && <span>+{candidates.length - 5}</span>}
            </div>
            <div className="upload-actions">
              <label className="upload-button"><UploadCloud size={15}/>添加简历<input type="file" multiple accept=".txt,.pdf,.docx" onChange={handleFiles}/></label>
              <button onClick={() => setCandidates(demo?.candidates || [])}><RefreshCw size={14}/>恢复演示集</button>
              {candidates.length > 0 && <button onClick={() => setCandidates([])}><X size={14}/>清空</button>}
              {candidates.length > 0 && <button onClick={() => saveCurrentCandidates().catch(err => setError(err.message))}><Database size={14}/>存入候选人库</button>}
            </div>
            <p className="privacy-note"><ShieldCheck size={15}/>姓名、联系方式及敏感属性不会进入匹配评分。</p>
          </div>
        </section>

        <section className="runbar">
          <div className="mode-switch">
            <button className={mode === 'rules' ? 'active' : ''} onClick={() => { setMode('rules'); setParsedJob(null); setCriteriaConfirmed(false) }}><RefreshCw size={15}/>快速规则模式</button>
            <button className={mode === 'adaptive' ? 'active' : ''} onClick={() => { setMode('adaptive'); setParsedJob(null); setCriteriaConfirmed(false) }}><Sparkles size={15}/>自适应Agent模式</button>
            <button className={mode === 'ollama' ? 'active' : ''} onClick={() => { setMode('ollama'); setParsedJob(null); setCriteriaConfirmed(false) }}><Bot size={15}/>Ollama智能模式</button>
          </div>
          <div className="run-action">
            {loading && <div className="task-progress" aria-live="polite">
              <div><span>{taskStageLabels[taskStage] || '智能体正在分析'}</span><b>{taskProgress}%</b></div>
              <progress max="100" value={taskProgress}/>
            </div>}
            <button className="primary" onClick={runAnalysis} disabled={loading || !demo || candidates.length === 0 || !criteriaConfirmed}>
              {loading ? <LoaderCircle className="spin" size={18}/> : <Play size={18}/>} {loading ? `分析中 ${taskProgress}%` : '开始证据化匹配'}
            </button>
          </div>
        </section>

        {error && <div className="error-box"><AlertTriangle size={18}/>{error}</div>}

        {!result && !error && (
          <section className="empty-state">
            <div className="empty-visual"><SearchCheck size={48}/></div>
            <h3>准备开始第一次匹配</h3>
            <p>快速模式约数秒完成；自适应与Ollama模式在后台调用本地模型，页面会展示当前分析阶段。</p>
            <div className="flow-line"><span>JD解析</span><ArrowRight/><span>证据抽取</span><ArrowRight/><span>透明评分</span><ArrowRight/><span>冲突复核</span></div>
          </section>
        )}

        {result && (
          <>
            <section className="metrics-grid">
              <Metric icon={<UsersRound/>} label="候选人" value={result.ranking.length}/>
              <Metric icon={<CheckCircle2/>} label="建议复核" value={result.ranking.filter(x => x.recommendation === 'recommended').length}/>
              <Metric icon={<AlertTriangle/>} label="冲突项" value={result.ranking.reduce((sum, x) => sum + x.findings.length, 0)}/>
              <Metric icon={<Activity/>} label="端到端耗时" value={`${(result.elapsed_ms / 1000).toFixed(1)}s`}/>
            </section>

            <section className="results-layout">
              <div className="panel ranking-panel">
                <div className="panel-heading"><div><span className="step">03</span><h2>候选人排序</h2></div><span className="counter">不自动淘汰</span></div>
                <div className="ranking-list">
                  {result.ranking.map((item, index) => {
                    const meta = recommendationMeta[item.recommendation]
                    return <button key={item.candidate_id} className={`rank-row ${selected?.candidate_id === item.candidate_id ? 'selected' : ''}`} onClick={() => { setSelectedId(item.candidate_id); setDecisionStatus('') }}>
                      <span className="rank-number">{String(index + 1).padStart(2, '0')}</span>
                      <span className="avatar"><CircleUserRound/></span>
                      <span className="rank-info"><b>{item.display_name}</b><small>{item.strengths.length}项强证据 · {item.gaps.length}项缺口</small></span>
                      <span className={`pill ${meta.className}`}>{meta.label}</span>
                      <span className="score"><b>{item.score.toFixed(1)}</b><small>匹配分</small></span>
                    </button>
                  })}
                </div>
              </div>

              {selected && <div className="panel evidence-panel">
                <div className="detail-head"><div><p className="eyebrow">EVIDENCE REVIEW</p><h2>{selected.display_name}</h2></div><div className="confidence">置信度 <b>{Math.round(selected.confidence * 100)}%</b></div></div>
                <div className="detail-summary"><div><b>{selected.score.toFixed(1)}</b><span>综合匹配分</span></div><p>分数由硬性条件、证据强度和技能覆盖计算，模型不直接决定最终分数。</p></div>
                <h3>要求与证据</h3>
                <div className="criteria-list">
                  {selected.criteria.map(item => {
                    const [label, cls] = statusMeta[item.status]
                    return <div className="criterion" key={item.requirement_id}>
                      <div><span className={`status-icon ${cls}`}/><b>{item.requirement_text}</b><span className={`tiny-pill ${cls}`}>{label}</span></div>
                      <p>{item.explanation}</p>
                    </div>
                  })}
                </div>
                {selected.findings.length > 0 && <><h3>复核提示</h3><div className="finding-list">{selected.findings.map((finding, idx) => <div className={`finding ${finding.severity}`} key={`${finding.code}-${idx}`}><AlertTriangle size={15}/><span>{finding.message}</span></div>)}</div></>}
                <div className="decision-box"><div><b>人工决策</b><span>{decisionStatus || '系统建议仅供参考，请记录最终处理意见'}</span></div><div><button className="advance" onClick={() => saveDecision('advance')}>进入下一轮</button><button onClick={() => saveDecision('hold')}>保持观察</button><button className="decline" onClick={() => saveDecision('not_advance')}>暂不推进</button></div></div>
              </div>}
            </section>

            <section className="panel trace-panel">
              <button className="trace-heading" onClick={() => setExpandedTrace(!expandedTrace)}><div><Bot size={19}/><b>Multi-Agent 执行轨迹</b><span>{result.traces.length}个节点完成</span></div>{expandedTrace ? <ChevronUp/> : <ChevronDown/>}</button>
              {expandedTrace && <div className="trace-grid">{result.traces.map((trace, idx) => <div className="trace-card" key={`${trace.node}-${idx}`}><span>{idx + 1}</span><div><b>{trace.node}</b><p>{trace.detail}</p><small>{trace.elapsed_ms.toFixed(0)} ms · {trace.status}</small></div></div>)}</div>}
            </section>
          </>
        )}
        </> : activeView === 'batches' ? <BatchManagementView
          jobs={managedJobs}
          candidates={managedCandidates}
          batches={screeningBatches}
          onCreate={createScreeningBatch}
          onDecision={saveBatchDecision}
          onBack={() => setActiveView('workbench')}
        /> : <ManagementView
          view={activeView}
          jobs={managedJobs}
          candidates={managedCandidates}
          runs={runHistory}
          audit={auditRecords}
          onBack={() => setActiveView('workbench')}
        />}
      </main>
    </div>
  )
}

function Metric({ icon, label, value }) {
  return <div className="metric-card"><span>{icon}</span><div><small>{label}</small><b>{value}</b></div></div>
}

function BatchManagementView({ jobs, candidates, batches, onCreate, onDecision, onBack }) {
  const [jobId, setJobId] = useState('')
  const [candidateIds, setCandidateIds] = useState([])
  const [batchMode, setBatchMode] = useState('rules')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const availableJobs = jobs.filter(item => item.status !== 'closed')
  const availableCandidates = candidates.filter(item => item.status !== 'archived')
  const statusLabels = {
    queued: '等待执行', running: '分析中', awaiting_review: '待人工复核', reviewed: '复核完成', failed: '执行失败',
  }

  function toggleCandidate(candidateId) {
    setCandidateIds(current => current.includes(candidateId)
      ? current.filter(item => item !== candidateId)
      : [...current, candidateId])
  }

  async function submitBatch(event) {
    event.preventDefault()
    setBusy(true)
    setMessage('批次正在执行，完成后会自动刷新结果。')
    try {
      const batch = await onCreate(jobId, candidateIds, batchMode)
      setMessage(batch.status === 'failed' ? `执行失败：${batch.error || '请查看日志'}` : '分析完成，请逐项进行人工复核。')
      setCandidateIds([])
    } catch (err) {
      setMessage(err.message)
    } finally {
      setBusy(false)
    }
  }

  return <section className="management-page batch-page">
    <header><div><p className="eyebrow">SCREENING WORKFLOW</p><h1>筛选批次</h1><p>从已保存岗位和候选人库发起分析，并持续记录人工处理结果。</p></div><button onClick={onBack}>返回匹配工作台</button></header>
    <form className="batch-builder panel" onSubmit={submitBatch}>
      <div><label>选择岗位<select value={jobId} onChange={event => setJobId(event.target.value)} required><option value="">请选择</option>{availableJobs.map(job => <option key={job.id} value={job.id}>{job.title} · {job.status}</option>)}</select></label></div>
      <div><label>分析模式<select value={batchMode} onChange={event => setBatchMode(event.target.value)}><option value="rules">快速规则模式</option><option value="adaptive">自适应 Agent</option><option value="ollama">完整 Ollama</option></select></label></div>
      <div className="batch-candidate-picker"><b>选择候选人（{candidateIds.length}/50）</b><div>{availableCandidates.map(candidate => <label key={candidate.id}><input type="checkbox" checked={candidateIds.includes(candidate.id)} onChange={() => toggleCandidate(candidate.id)}/><span>{candidate.display_name}</span><small>{candidate.external_ref || candidate.id}</small></label>)}</div></div>
      <button className="primary" disabled={busy || !jobId || candidateIds.length === 0}>{busy ? <LoaderCircle className="spin" size={17}/> : <Play size={17}/>}创建并执行筛选批次</button>
      {message && <p className="batch-message">{message}</p>}
    </form>
    <div className="batch-list">
      {batches.length === 0 && <div className="empty-state compact"><ListChecks size={36}/><h3>暂无筛选批次</h3><p>选择一个岗位和候选人后创建首个可追踪批次。</p></div>}
      {batches.map(batch => <article className="batch-card panel" key={batch.id}>
        <div className="batch-card-head"><div><b>{batch.job_title}</b><span>{batch.id} · {batch.mode}</span></div><span className={`tiny-pill ${batch.status === 'failed' ? 'danger' : batch.status === 'reviewed' ? 'success' : 'warning'}`}>{statusLabels[batch.status]}</span></div>
        <div className="batch-progress"><span>{batch.reviewed_count}/{batch.candidate_count} 已复核</span><progress max={batch.candidate_count} value={batch.reviewed_count}/></div>
        {batch.error && <p className="batch-error">{batch.error}</p>}
        <div className="batch-items">{batch.items.map(item => <div key={item.candidate_id}><div><b>{item.display_name}</b><small>{item.stage}</small></div>{item.decision ? <span className="tiny-pill success">{item.decision}</span> : batch.status === 'awaiting_review' ? <div className="batch-decisions"><button onClick={() => onDecision(batch.id, item.candidate_id, 'advance')}>进入下一轮</button><button onClick={() => onDecision(batch.id, item.candidate_id, 'hold')}>保持观察</button><button onClick={() => onDecision(batch.id, item.candidate_id, 'not_advance')}>暂不推进</button></div> : <span className="tiny-pill neutral">等待分析</span>}</div>)}</div>
        <small>更新于 {new Date(batch.updated_at).toLocaleString()} {batch.run_id ? `· ${batch.run_id}` : ''}</small>
      </article>)}
    </div>
  </section>
}

function ManagementView({ view, jobs, candidates, runs, audit, onBack }) {
  const meta = {
    jobs: ['岗位管理', '统一维护草稿、在招与已关闭岗位'],
    candidates: ['候选人库', '保存候选人材料与招聘流程状态'],
    runs: ['运行记录', '查看历史匹配批次和评测结果'],
    audit: ['审计记录', '追踪登录、数据写入和人工决策操作'],
  }[view]
  const items = { jobs, candidates, runs, audit }[view] || []
  return <section className="management-page">
    <header><div><p className="eyebrow">ENTERPRISE DATA WORKSPACE</p><h1>{meta[0]}</h1><p>{meta[1]}</p></div><button onClick={onBack}>返回匹配工作台</button></header>
    <div className="management-list">
      {items.length === 0 && <div className="empty-state compact"><Database size={36}/><h3>暂无记录</h3><p>从匹配工作台保存数据后会显示在这里。</p></div>}
      {view === 'jobs' && items.map(item => <article key={item.id}><div><b>{item.title}</b><span>{item.id}</span></div><span className="tiny-pill neutral">{item.status}</span><p>{item.description.slice(0, 180)}</p><small>更新于 {new Date(item.updated_at).toLocaleString()}</small></article>)}
      {view === 'candidates' && items.map(item => <article key={item.id}><div><b>{item.display_name}</b><span>{item.external_ref || item.id}</span></div><span className="tiny-pill neutral">{item.status}</span><p>{item.resume_text.slice(0, 180)}</p><small>更新于 {new Date(item.updated_at).toLocaleString()}</small></article>)}
      {view === 'runs' && items.map(item => <article key={item.run_id}><div><b>{item.job_title}</b><span>{item.run_id}</span></div><span className="tiny-pill neutral">{item.mode}</span><p>{item.candidate_count} 名候选人</p><small>{new Date(item.created_at).toLocaleString()}</small></article>)}
      {view === 'audit' && items.map(item => <article key={item.id}><div><b>{item.actor_username} · {item.action}</b><span>{item.resource_type}</span></div><p>{item.resource_id || '系统级操作'}</p><small>{new Date(item.created_at).toLocaleString()}</small></article>)}
    </div>
  </section>
}

export default App
