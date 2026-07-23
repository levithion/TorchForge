"use client";

import {
  Activity,
  AlertCircle,
  ArrowRight,
  Braces,
  Check,
  ChevronRight,
  Circle,
  Code2,
  Cpu,
  FileCode2,
  FileText,
  Gauge,
  Layers3,
  LoaderCircle,
  Menu,
  PanelLeftClose,
  Play,
  RefreshCw,
  Search,
  Server,
  Sparkles,
  UploadCloud,
  X,
  Zap,
} from "lucide-react";
import { ChangeEvent, DragEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

type StageName = "extract" | "parse" | "compile" | "validate";

type Paper = {
  id: string;
  title: string;
  source: string;
  pageCount: number;
  status: string;
  warnings: string[];
  errors: string[];
  stages: Record<StageName, boolean>;
  availableArtifacts: string[];
  visionModel: string | null;
  codeModel: string | null;
  validation: {
    status: string;
    device: string;
    attempt_count: number;
    output_shapes: number[][];
    architecture_profile: string | null;
    conformance_passed: boolean;
  } | null;
};

type Health = {
  status: string;
  device: string;
  ollama: { ready: boolean; models: string[] };
  defaults: { visionModel: string; codeModel: string };
};

const stages: {
  id: StageName;
  number: string;
  title: string;
  short: string;
  description: string;
  artifact: string;
  icon: typeof FileText;
}[] = [
  {
    id: "extract",
    number: "01",
    title: "Extract paper",
    short: "Extract",
    description: "Text, metadata & figures",
    artifact: "text",
    icon: FileText,
  },
  {
    id: "parse",
    number: "02",
    title: "Map architecture",
    short: "Parse",
    description: "Vision-to-topology analysis",
    artifact: "topology",
    icon: Layers3,
  },
  {
    id: "compile",
    number: "03",
    title: "Generate module",
    short: "Compile",
    description: "Validated PyTorch source",
    artifact: "code",
    icon: Code2,
  },
  {
    id: "validate",
    number: "04",
    title: "Prove runtime",
    short: "Validate",
    description: "CUDA, MPS or CPU execution",
    artifact: "validation",
    icon: Gauge,
  },
];

function apiBase(): string {
  if (process.env.NEXT_PUBLIC_TORCHFORGE_API_URL) {
    return process.env.NEXT_PUBLIC_TORCHFORGE_API_URL.replace(/\/$/, "");
  }
  return "http://127.0.0.1:8000";
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase()}${path}`, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // The plain fallback above remains useful.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

function compactId(id: string): string {
  const suffix = id.match(/[a-f0-9]{12}$/)?.[0];
  return suffix ? suffix.slice(0, 7).toUpperCase() : id.slice(0, 7).toUpperCase();
}

function stageProgress(paper: Paper): number {
  return stages.filter((stage) => paper.stages[stage.id]).length;
}

function statusLabel(paper: Paper): string {
  const completed = stageProgress(paper);
  if (completed === 4) return paper.validation?.status === "repaired" ? "Repaired" : "Validated";
  if (paper.errors.length) return "Needs attention";
  return `Phase ${Math.max(1, completed)} complete`;
}

export function TorchForgeApp() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [engineOnline, setEngineOnline] = useState(false);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [runningStage, setRunningStage] = useState<StageName | null>(null);
  const [query, setQuery] = useState("");
  const [notice, setNotice] = useState<{ kind: "error" | "success"; text: string } | null>(null);
  const [artifact, setArtifact] = useState<{ name: string; content: string } | null>(null);
  const [artifactLoading, setArtifactLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const selected = papers.find((paper) => paper.id === selectedId) ?? null;
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return papers;
    return papers.filter(
      (paper) =>
        paper.title.toLowerCase().includes(needle) ||
        paper.source.toLowerCase().includes(needle) ||
        paper.id.toLowerCase().includes(needle),
    );
  }, [papers, query]);

  const refresh = useCallback(async () => {
    const [healthResult, paperResult] = await Promise.allSettled([
      api<Health>("/api/health"),
      api<{ papers: Paper[] }>("/api/papers"),
    ]);
    if (healthResult.status === "fulfilled") {
      setHealth(healthResult.value);
      setEngineOnline(true);
    } else {
      setEngineOnline(false);
    }
    if (paperResult.status === "fulfilled") {
      setPapers(paperResult.value.papers);
      setSelectedId((current) => current ?? paperResult.value.papers[0]?.id ?? null);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => window.clearTimeout(timer);
  }, [refresh]);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 5200);
    return () => window.clearTimeout(timer);
  }, [notice]);

  async function upload(file: File) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setNotice({ kind: "error", text: "TorchForge accepts PDF papers only." });
      return;
    }
    setUploading(true);
    try {
      const paper = await api<Paper>("/api/papers", {
        method: "POST",
        headers: {
          "Content-Type": "application/pdf",
          "X-Filename": file.name,
        },
        body: file,
      });
      setPapers((current) => [paper, ...current.filter((item) => item.id !== paper.id)]);
      setSelectedId(paper.id);
      setNotice({ kind: "success", text: "Paper extracted. Architecture mapping is ready to run." });
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : "Upload failed." });
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) void upload(file);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) void upload(file);
  }

  async function runStage(stage: StageName) {
    if (!selected || stage === "extract") return;
    setRunningStage(stage);
    try {
      const updated = await api<Paper>(`/api/papers/${encodeURIComponent(selected.id)}/${stage}`, {
        method: "POST",
      });
      setPapers((current) => current.map((paper) => (paper.id === updated.id ? updated : paper)));
      setNotice({
        kind: "success",
        text: `${stages.find((item) => item.id === stage)?.title} completed successfully.`,
      });
    } catch (error) {
      setNotice({
        kind: "error",
        text: error instanceof Error ? error.message : `${stage} failed.`,
      });
    } finally {
      setRunningStage(null);
    }
  }

  async function openArtifact(name: string) {
    if (!selected) return;
    setArtifactLoading(true);
    setArtifact({ name, content: "" });
    try {
      const response = await fetch(
        `${apiBase()}/api/papers/${encodeURIComponent(selected.id)}/artifacts/${name}`,
      );
      if (!response.ok) throw new Error("Artifact is not available yet.");
      setArtifact({ name, content: await response.text() });
    } catch (error) {
      setArtifact(null);
      setNotice({ kind: "error", text: error instanceof Error ? error.message : "Could not open artifact." });
    } finally {
      setArtifactLoading(false);
    }
  }

  function choosePaper(paper: Paper) {
    setSelectedId(paper.id);
    setSidebarOpen(false);
  }

  const nextStage = selected
    ? stages.find((stage) => stage.id !== "extract" && !selected.stages[stage.id])
    : null;

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Zap size={19} strokeWidth={2.4} /></div>
          <div>
            <span className="brand-name">TorchForge</span>
            <span className="brand-version">LOCAL STUDIO</span>
          </div>
          <button className="icon-button sidebar-close" onClick={() => setSidebarOpen(false)} aria-label="Close navigation">
            <PanelLeftClose size={19} />
          </button>
        </div>

        <nav className="nav-list" aria-label="Main navigation">
          <button className="nav-item active"><Braces size={18} /><span>Forge</span><span className="nav-dot" /></button>
          <button className="nav-item" onClick={() => document.getElementById("library")?.scrollIntoView({ behavior: "smooth" })}>
            <FileCode2 size={18} /><span>Paper library</span><span className="nav-count">{papers.length}</span>
          </button>
          <button className="nav-item" onClick={() => void refresh()}><Activity size={18} /><span>Environment</span></button>
        </nav>

        <div className="sidebar-section">
          <p className="eyebrow">RECENT PAPERS</p>
          <div className="recent-list">
            {papers.slice(0, 4).map((paper) => (
              <button
                key={paper.id}
                className={`recent-paper ${selectedId === paper.id ? "selected" : ""}`}
                onClick={() => choosePaper(paper)}
              >
                <FileText size={16} />
                <span>{paper.title}</span>
              </button>
            ))}
            {!papers.length && <p className="sidebar-empty">Your processed papers will appear here.</p>}
          </div>
        </div>

        <div className="engine-card">
          <div className="engine-card-head">
            <span className={`status-light ${engineOnline ? "online" : ""}`} />
            <span>{engineOnline ? "Engine online" : "Engine offline"}</span>
          </div>
          <p>{engineOnline ? `Running on ${health?.device.toUpperCase()}` : "Start the local TorchForge API to process papers."}</p>
          <div className="engine-meta">
            <Cpu size={15} />
            <span>{health?.ollama.ready ? "Ollama connected" : "Ollama unavailable"}</span>
          </div>
        </div>
      </aside>

      {sidebarOpen && <button className="sidebar-scrim" onClick={() => setSidebarOpen(false)} aria-label="Close navigation" />}

      <main className="main">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setSidebarOpen(true)} aria-label="Open navigation"><Menu size={21} /></button>
          <div className="breadcrumbs">
            <span>Workspace</span><ChevronRight size={14} /><strong>Research forge</strong>
          </div>
          <div className="topbar-actions">
            <div className={`connection-pill ${engineOnline ? "connected" : ""}`}>
              <span className="status-light online" />
              <span>{engineOnline ? `${health?.device.toUpperCase()} ready` : "Local engine offline"}</span>
            </div>
            <button className="icon-button" onClick={() => void refresh()} aria-label="Refresh workspace"><RefreshCw size={18} /></button>
          </div>
        </header>

        <div className="content">
          <section className="hero">
            <div className="hero-copy">
              <div className="kicker"><Sparkles size={15} /> PAPER → RUNNABLE MODEL</div>
              <h1>Forge research into<br /><span>working PyTorch.</span></h1>
              <p>Drop in a transformer paper. TorchForge extracts the architecture, writes an implementation, and proves it on your hardware.</p>
            </div>
            <div className="hero-stats" aria-label="Workspace statistics">
              <div><strong>{papers.length}</strong><span>Papers</span></div>
              <div><strong>{papers.filter((paper) => paper.stages.validate).length}</strong><span>Validated</span></div>
              <div><strong>{health?.device?.toUpperCase() ?? "—"}</strong><span>Runtime</span></div>
            </div>
          </section>

          {!engineOnline && !loading && (
            <section className="offline-banner">
              <div className="offline-icon"><Server size={20} /></div>
              <div>
                <strong>Connect the local TorchForge engine</strong>
                <p>Start <code>uv run torchforge serve</code> in the project folder, then refresh this page.</p>
              </div>
              <button onClick={() => void refresh()}>Try again <RefreshCw size={15} /></button>
            </section>
          )}

          <section className="workbench">
            <div
              className={`dropzone ${dragging ? "dragging" : ""} ${uploading ? "uploading" : ""}`}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => engineOnline && !uploading && fileInput.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {
                if ((event.key === "Enter" || event.key === " ") && engineOnline) fileInput.current?.click();
              }}
              aria-label="Upload a research paper PDF"
            >
              <input ref={fileInput} type="file" accept=".pdf,application/pdf" onChange={onFileChange} hidden />
              <div className="drop-orbit">
                {uploading ? <LoaderCircle className="spin" size={28} /> : <UploadCloud size={28} />}
              </div>
              <div className="drop-copy">
                <strong>{uploading ? "Extracting your paper…" : "Drop a research paper here"}</strong>
                <span>{uploading ? "Reading text, metadata, and figure pages" : "or click to choose a PDF · up to 100 MB"}</span>
              </div>
              <button className="upload-button" disabled={!engineOnline || uploading}>
                {uploading ? "Processing" : "Choose PDF"} {!uploading && <ArrowRight size={16} />}
              </button>
            </div>

            <div className="runtime-panel">
              <div className="panel-heading">
                <span>LOCAL RUNTIME</span>
                <span className={`mini-status ${engineOnline ? "good" : ""}`}>{engineOnline ? "READY" : "OFFLINE"}</span>
              </div>
              <div className="runtime-row">
                <div className="runtime-icon"><Cpu size={19} /></div>
                <div><span>Compute device</span><strong>{health?.device?.toUpperCase() ?? "Not detected"}</strong></div>
                {health && <Check size={17} className="success-icon" />}
              </div>
              <div className="runtime-row">
                <div className="runtime-icon violet"><Sparkles size={19} /></div>
                <div><span>Vision engine</span><strong>{health?.defaults.visionModel ?? "LLaVA"}</strong></div>
                <span className={`status-light ${health?.ollama.ready ? "online" : ""}`} />
              </div>
              <div className="runtime-row">
                <div className="runtime-icon amber"><Code2 size={19} /></div>
                <div><span>Code engine</span><strong>{health?.defaults.codeModel ?? "Qwen Coder"}</strong></div>
                <span className={`status-light ${health?.ollama.ready ? "online" : ""}`} />
              </div>
            </div>
          </section>

          {selected && (
            <section className="selected-work">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">ACTIVE FORGE</span>
                  <h2>{selected.title}</h2>
                  <p>{selected.source} · {selected.pageCount} {selected.pageCount === 1 ? "page" : "pages"} · ID {compactId(selected.id)}</p>
                </div>
                <div className={`paper-status status-${stageProgress(selected)}`}>
                  <span className="status-light online" />{statusLabel(selected)}
                </div>
              </div>

              <div className="pipeline">
                {stages.map((stage, index) => {
                  const done = selected.stages[stage.id];
                  const previousDone = index === 0 || selected.stages[stages[index - 1].id];
                  const isRunning = runningStage === stage.id;
                  const canRun = stage.id !== "extract" && previousDone && !done && runningStage === null;
                  const Icon = stage.icon;
                  return (
                    <div className={`stage-card ${done ? "done" : ""} ${isRunning ? "running" : ""}`} key={stage.id}>
                      <div className="stage-top">
                        <span className="stage-number">{stage.number}</span>
                        <div className="stage-state">
                          {done ? <Check size={15} /> : isRunning ? <LoaderCircle className="spin" size={15} /> : <Circle size={12} />}
                          <span>{done ? "DONE" : isRunning ? "RUNNING" : "PENDING"}</span>
                        </div>
                      </div>
                      <div className="stage-icon"><Icon size={21} /></div>
                      <h3>{stage.title}</h3>
                      <p>{stage.description}</p>
                      {done && selected.availableArtifacts.includes(stage.artifact) ? (
                        <button className="stage-action artifact-action" onClick={() => void openArtifact(stage.artifact)}>
                          View output <ArrowRight size={14} />
                        </button>
                      ) : canRun ? (
                        <button className="stage-action run-action" onClick={() => void runStage(stage.id)}>
                          <Play size={13} fill="currentColor" /> Run phase
                        </button>
                      ) : (
                        <span className="stage-action locked">{stage.id === "extract" ? "Uploaded" : "Awaiting prior phase"}</span>
                      )}
                    </div>
                  );
                })}
              </div>

              {nextStage && (
                <div className="next-action">
                  <div>
                    <span>NEXT RECOMMENDED ACTION</span>
                    <strong>{nextStage.title}</strong>
                    <p>{nextStage.description}. This may take several minutes on CPU.</p>
                  </div>
                  <button onClick={() => void runStage(nextStage.id)} disabled={runningStage !== null || !health?.ollama.ready}>
                    {runningStage === nextStage.id ? <LoaderCircle className="spin" size={17} /> : <Play size={15} fill="currentColor" />}
                    {runningStage === nextStage.id ? "Running…" : `Run ${nextStage.short}`}
                  </button>
                </div>
              )}

              {selected.validation && (
                <div className="validation-strip">
                  <div>
                    <Check size={18} />
                    <span>
                      {selected.validation.architecture_profile
                        ? "Architecture verified"
                        : "Runtime verified"}
                    </span>
                  </div>
                  {selected.validation.architecture_profile && (
                    <div>
                      <span>Profile</span>
                      <strong>{selected.validation.architecture_profile.replaceAll("_", " ").toUpperCase()}</strong>
                    </div>
                  )}
                  <div><span>Device</span><strong>{selected.validation.device.toUpperCase()}</strong></div>
                  <div><span>Attempts</span><strong>{selected.validation.attempt_count}</strong></div>
                  <div><span>Output</span><strong>{selected.validation.output_shapes.map((shape) => `[${shape.join(", ")}]`).join(" · ")}</strong></div>
                </div>
              )}
            </section>
          )}

          <section className="library" id="library">
            <div className="library-head">
              <div>
                <span className="eyebrow">PAPER LIBRARY</span>
                <h2>Your research workspace</h2>
              </div>
              <label className="search">
                <Search size={16} />
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search papers…" />
              </label>
            </div>

            <div className="paper-list">
              {loading && (
                <div className="loading-state"><LoaderCircle className="spin" size={22} /> Loading workspace…</div>
              )}
              {!loading && !filtered.length && (
                <div className="empty-state">
                  <div><FileText size={25} /></div>
                  <strong>{papers.length ? "No papers match your search" : "Your forge is ready"}</strong>
                  <p>{papers.length ? "Try a different title or identifier." : "Upload your first transformer paper to begin."}</p>
                </div>
              )}
              {filtered.map((paper) => (
                <button className={`paper-row ${selectedId === paper.id ? "active" : ""}`} key={paper.id} onClick={() => choosePaper(paper)}>
                  <div className="paper-file-icon"><FileText size={19} /></div>
                  <div className="paper-main">
                    <strong>{paper.title}</strong>
                    <span>{paper.source} · {paper.pageCount} pages · {compactId(paper.id)}</span>
                  </div>
                  <div className="mini-pipeline" aria-label={`${stageProgress(paper)} of 4 phases complete`}>
                    {stages.map((stage) => <span className={paper.stages[stage.id] ? "filled" : ""} key={stage.id} />)}
                  </div>
                  <span className={`row-status row-status-${stageProgress(paper)}`}>{statusLabel(paper)}</span>
                  <ChevronRight size={17} className="row-chevron" />
                </button>
              ))}
            </div>
          </section>

          <footer>
            <span><Zap size={14} /> TorchForge runs locally. Your research stays on your machine.</span>
            <span>Phase 1–4 pipeline · v0.4</span>
          </footer>
        </div>
      </main>

      {(artifact || artifactLoading) && (
        <div className="artifact-overlay" role="dialog" aria-modal="true" aria-label="Artifact viewer">
          <button className="artifact-scrim" onClick={() => setArtifact(null)} aria-label="Close artifact" />
          <section className="artifact-drawer">
            <header>
              <div>
                <span className="eyebrow">GENERATED ARTIFACT</span>
                <h2>{artifact?.name ?? "Loading"}<span>.</span>{artifact?.name === "code" ? "py" : artifact?.name === "text" ? "md" : "json"}</h2>
              </div>
              <button className="icon-button" onClick={() => setArtifact(null)} aria-label="Close artifact"><X size={20} /></button>
            </header>
            <div className="artifact-meta">
              <span>{selected?.title}</span>
              <span>Read only</span>
            </div>
            {artifactLoading ? (
              <div className="artifact-loading"><LoaderCircle className="spin" size={24} /> Loading artifact…</div>
            ) : (
              <pre><code>{artifact?.content}</code></pre>
            )}
          </section>
        </div>
      )}

      {notice && (
        <div className={`toast ${notice.kind}`}>
          {notice.kind === "success" ? <Check size={18} /> : <AlertCircle size={18} />}
          <span>{notice.text}</span>
          <button onClick={() => setNotice(null)} aria-label="Dismiss notification"><X size={16} /></button>
        </div>
      )}
    </div>
  );
}
