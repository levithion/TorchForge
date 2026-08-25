"use client";

import Editor, { DiffEditor } from "@monaco-editor/react";
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  Background,
  Controls,
  Edge,
  EdgeChange,
  MarkerType,
  MiniMap,
  Node,
  NodeChange,
  ReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  Activity,
  AlertCircle,
  Archive,
  Boxes,
  Check,
  ChevronDown,
  CircleStop,
  Clipboard,
  Code2,
  Columns3,
  Copy,
  Cpu,
  Download,
  FileArchive,
  FileCode2,
  FileDown,
  FileText,
  Gauge,
  GitCompareArrows,
  Layers3,
  LoaderCircle,
  Menu,
  PanelLeftClose,
  Pencil,
  Play,
  Plus,
  RefreshCw,
  Save,
  Search,
  Server,
  Settings2,
  Sparkles,
  Tags,
  Trash2,
  UploadCloud,
  X,
  Zap,
} from "lucide-react";
import {
  ChangeEvent,
  DragEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

type StageName = "extract" | "parse" | "compile" | "validate";
type RunnableStage = Exclude<StageName, "extract">;
type WorkspaceTab = "overview" | "topology" | "evidence" | "code" | "validation" | "compare";

type ValidationSummary = {
  status: string;
  device: string;
  attempt_count: number;
  output_shapes: number[][];
  architecture_profile: string | null;
  conformance_passed: boolean;
};

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
  validation: ValidationSummary | null;
  tags: string[];
  archived: boolean;
  createdAt: string | null;
  updatedAt: string | null;
};

type Health = {
  status: string;
  device: string;
  ollama: { ready: boolean; models: string[] };
  defaults: { visionModel: string; codeModel: string };
  capabilities?: Record<string, boolean>;
};

type StageOptions = {
  vision_model: string;
  code_model: string;
  device: "auto" | "cpu" | "cuda" | "mps";
  max_images: number;
  context_window: number;
  max_output_tokens: number;
  max_text_chars: number;
  max_repairs: number;
  timeout: number;
};

type Job = {
  id: string;
  paperId: string;
  stages: RunnableStage[];
  stage: RunnableStage | null;
  status: "queued" | "running" | "cancelling" | "completed" | "failed" | "cancelled";
  progress: number;
  logs: { time: string; message: string }[];
  error: string | null;
  durationMs: number | null;
  createdAt: string;
};

type TensorSpec = {
  name: string;
  shape?: (number | null)[] | null;
  dtype?: string | null;
  description?: string | null;
};

type LayerSpec = {
  id: string;
  layer_type: string;
  inputs: string[];
  input_shape?: (number | null)[] | null;
  output_shape?: (number | null)[] | null;
  parameters: Record<string, unknown>;
  description?: string | null;
  confidence: number;
};

type ConnectionSpec = {
  source: string;
  target: string;
  kind: "sequential" | "skip" | "residual" | "concat" | "cross_attention" | "other";
  description?: string | null;
};

type Topology = {
  schema_version: "1.0";
  architecture_name: string;
  task?: string | null;
  inputs: TensorSpec[];
  layers: LayerSpec[];
  connections: ConnectionSpec[];
  outputs: TensorSpec[];
  assumptions: string[];
  source_images: string[];
  overall_confidence: number;
};

type Evidence = {
  sourceAvailable: boolean;
  images: { path: string; page: number | null; kind: string }[];
  visionSources: string[];
};

type PerformanceMetrics = {
  latency_ms_mean: number;
  latency_ms_p50: number;
  latency_ms_p95: number;
  throughput_samples_per_sec: number;
  measured_forward_passes: number;
  peak_memory_bytes?: number | null;
  estimated_flops?: number | null;
};

type ValidationDetail = {
  status: string;
  device: string;
  class_name?: string | null;
  constructor_kwargs?: Record<string, unknown>;
  input_shapes?: number[][];
  output_shapes?: number[][];
  architecture_profile?: string | null;
  conformance_checks?: { name: string; passed: boolean; detail: string }[];
  attempts?: { attempt: number; code_path: string; succeeded: boolean; error?: string | null }[];
  performance?: PerformanceMetrics | null;
};

type Revision = { artifact: string; path: string; created_at: string };

const defaultOptions: StageOptions = {
  vision_model: "llava",
  code_model: "qwen2.5-coder:3b",
  device: "auto",
  max_images: 8,
  context_window: 8192,
  max_output_tokens: 4096,
  max_text_chars: 6000,
  max_repairs: 2,
  timeout: 600,
};

const stageDefinitions: {
  id: StageName;
  number: string;
  title: string;
  description: string;
  artifact: string;
  icon: typeof FileText;
}[] = [
  { id: "extract", number: "01", title: "Extract", description: "Text, metadata & figures", artifact: "text", icon: FileText },
  { id: "parse", number: "02", title: "Map", description: "Vision topology", artifact: "topology", icon: Layers3 },
  { id: "compile", number: "03", title: "Generate", description: "PyTorch source", artifact: "code", icon: Code2 },
  { id: "validate", number: "04", title: "Validate", description: "Runtime proof", artifact: "validation", icon: Gauge },
];

const tabs: { id: WorkspaceTab; label: string; icon: typeof FileText }[] = [
  { id: "overview", label: "Overview", icon: Activity },
  { id: "topology", label: "Topology", icon: Boxes },
  { id: "evidence", label: "Evidence", icon: FileText },
  { id: "code", label: "Code", icon: Code2 },
  { id: "validation", label: "Validation", icon: Gauge },
  { id: "compare", label: "Compare", icon: GitCompareArrows },
];

function apiBase(): string {
  return (process.env.NEXT_PUBLIC_TORCHFORGE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase()}${path}`, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Preserve the status fallback for non-JSON responses.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

async function textArtifact(paperId: string, name: string): Promise<string> {
  const response = await fetch(
    `${apiBase()}/api/papers/${encodeURIComponent(paperId)}/artifacts/${encodeURIComponent(name)}`,
  );
  if (!response.ok) throw new Error(`Could not load ${name} (${response.status}).`);
  return response.text();
}

function compactId(id: string): string {
  const suffix = id.match(/[a-f0-9]{12}$/)?.[0];
  return (suffix || id).slice(0, 7).toUpperCase();
}

function stageProgress(paper: Paper): number {
  return stageDefinitions.filter((stage) => paper.stages[stage.id]).length;
}

function nextStages(paper: Paper): RunnableStage[] {
  return stageDefinitions
    .filter((stage): stage is (typeof stageDefinitions)[number] & { id: RunnableStage } => (
      stage.id !== "extract" && !paper.stages[stage.id]
    ))
    .map((stage) => stage.id);
}

function formatDuration(milliseconds: number | null): string {
  if (!milliseconds) return "—";
  if (milliseconds < 1000) return `${milliseconds} ms`;
  return `${(milliseconds / 1000).toFixed(1)} s`;
}

function formatShape(shape?: (number | null)[] | null): string {
  return shape ? `[${shape.map((value) => value ?? "?").join(", ")}]` : "shape unknown";
}

function formatBytes(bytes?: number | null): string {
  if (!bytes || bytes <= 0) return "—";
  const units = ["B", "KiB", "MiB", "GiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 100 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

function formatFlops(flops?: number | null): string {
  if (!flops || flops <= 0) return "—";
  const units = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs"];
  let value = flops;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${value >= 100 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

function evidenceUrl(paperId: string, path: string): string {
  const encoded = path.split("/").map(encodeURIComponent).join("/");
  return `${apiBase()}/api/papers/${encodeURIComponent(paperId)}/evidence/${encoded}`;
}

export function TorchForgeApp() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [engineOnline, setEngineOnline] = useState(false);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [query, setQuery] = useState("");
  const [showArchived, setShowArchived] = useState(false);
  const [tab, setTab] = useState<WorkspaceTab>("overview");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [options, setOptions] = useState<StageOptions>(() => {
    if (typeof window === "undefined") return defaultOptions;
    const saved = window.localStorage.getItem("torchforge-stage-options");
    if (!saved) return defaultOptions;
    try {
      return { ...defaultOptions, ...(JSON.parse(saved) as Partial<StageOptions>) };
    } catch {
      window.localStorage.removeItem("torchforge-stage-options");
      return defaultOptions;
    }
  });
  const [notice, setNotice] = useState<{ kind: "error" | "success"; text: string } | null>(null);
  const [artifact, setArtifact] = useState<{ name: string; content: string; dirty: boolean } | null>(null);
  const [artifactLoading, setArtifactLoading] = useState(false);
  const [topology, setTopology] = useState<Topology | null>(null);
  const [topologyDirty, setTopologyDirty] = useState(false);
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null);
  const [graphNodes, setGraphNodes] = useState<Node[]>([]);
  const [graphEdges, setGraphEdges] = useState<Edge[]>([]);
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [validation, setValidation] = useState<ValidationDetail | null>(null);
  const [revisions, setRevisions] = useState<Revision[]>([]);
  const [selectedRevision, setSelectedRevision] = useState<{ path: string; content: string } | null>(null);
  const [compareId, setCompareId] = useState("");
  const [compareCode, setCompareCode] = useState("");
  const [tagDraft, setTagDraft] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const completedJobs = useRef<Set<string>>(new Set());

  const selected = papers.find((paper) => paper.id === selectedId) ?? null;
  const selectedLayer = topology?.layers.find((layer) => layer.id === selectedLayerId) ?? null;
  const activeJobs = jobs.filter((job) => ["queued", "running", "cancelling"].includes(job.status));
  const selectedJobs = jobs.filter((job) => job.paperId === selectedId);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return papers.filter((paper) => {
      if (!showArchived && paper.archived) return false;
      if (!needle) return true;
      return [paper.title, paper.source, paper.id, ...(paper.tags ?? [])]
        .join(" ")
        .toLowerCase()
        .includes(needle);
    });
  }, [papers, query, showArchived]);

  const refresh = useCallback(async () => {
    const [healthResult, paperResult, jobResult] = await Promise.allSettled([
      api<Health>("/api/health"),
      api<{ papers: Paper[] }>("/api/papers"),
      api<{ jobs: Job[] }>("/api/jobs"),
    ]);
    if (healthResult.status === "fulfilled") {
      setHealth(healthResult.value);
      setEngineOnline(true);
      setOptions((current) => ({
        ...current,
        vision_model: current.vision_model || healthResult.value.defaults.visionModel,
        code_model: current.code_model || healthResult.value.defaults.codeModel,
      }));
    } else {
      setEngineOnline(false);
    }
    if (paperResult.status === "fulfilled") {
      setPapers(paperResult.value.papers);
      setSelectedId((current) => current ?? paperResult.value.papers[0]?.id ?? null);
    }
    if (jobResult.status === "fulfilled") setJobs(jobResult.value.jobs);
    setLoading(false);
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => window.clearTimeout(timer);
  }, [refresh]);

  useEffect(() => {
    window.localStorage.setItem("torchforge-stage-options", JSON.stringify(options));
  }, [options]);

  useEffect(() => {
    let source: EventSource | null = null;
    let pollTimer: number | null = null;

    const ingestJobs = (nextJobs: Job[]) => {
      setJobs(nextJobs);
      const newlyCompleted = nextJobs.filter(
        (job) => job.status === "completed" && !completedJobs.current.has(job.id),
      );
      for (const job of newlyCompleted) completedJobs.current.add(job.id);
      if (newlyCompleted.length) void refresh();
    };

    const startPollingFallback = () => {
      if (pollTimer !== null) return;
      const tick = async () => {
        try {
          const result = await api<{ jobs: Job[] }>("/api/jobs");
          ingestJobs(result.jobs);
        } catch {
          // The health banner communicates a temporarily unavailable engine.
        }
      };
      pollTimer = window.setInterval(tick, activeJobs.length ? 900 : 4000);
    };

    if (typeof window !== "undefined" && "EventSource" in window) {
      source = new EventSource(`${apiBase()}/api/jobs/stream`);
      source.onmessage = (event) => {
        try {
          ingestJobs(JSON.parse(event.data) as Job[]);
        } catch {
          // Ignore malformed frames; the next snapshot will catch up.
        }
      };
      source.onerror = () => {
        // The stream closes when all jobs settle and reconnects automatically;
        // fall back to polling only if the engine is unreachable.
        if (!engineOnline) startPollingFallback();
      };
    } else {
      startPollingFallback();
    }

    return () => {
      source?.close();
      if (pollTimer !== null) window.clearInterval(pollTimer);
    };
  }, [activeJobs.length, engineOnline, refresh]);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 6000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setArtifact(null);
      setTopology(null);
      setEvidence(null);
      setValidation(null);
      setSelectedLayerId(null);
      setSelectedRevision(null);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [selectedId]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (!topology) {
        setGraphNodes([]);
        setGraphEdges([]);
        return;
      }
      const columns = Math.max(1, Math.ceil(Math.sqrt(topology.layers.length)));
      setGraphNodes(
        topology.layers.map((layer, index) => ({
          id: layer.id,
          position: { x: (index % columns) * 230, y: Math.floor(index / columns) * 145 },
          data: {
            label: `${layer.layer_type}\n${formatShape(layer.output_shape)}\n${Math.round(layer.confidence * 100)}%`,
          },
          className: layer.id === selectedLayerId ? "topology-node selected" : "topology-node",
        })),
      );
      setGraphEdges(
        topology.connections.map((connection, index) => ({
          id: `${connection.source}-${connection.target}-${index}`,
          source: connection.source,
          target: connection.target,
          label: connection.kind === "sequential" ? undefined : connection.kind,
          animated: connection.kind === "residual" || connection.kind === "skip",
          markerEnd: { type: MarkerType.ArrowClosed },
        })),
      );
    }, 0);
    return () => window.clearTimeout(timer);
  }, [topology, selectedLayerId]);

  async function upload(files: File[]) {
    const pdfs = files.filter((file) => file.name.toLowerCase().endsWith(".pdf"));
    if (!pdfs.length) {
      setNotice({ kind: "error", text: "TorchForge accepts PDF papers only." });
      return;
    }
    setUploading(pdfs.length);
    const uploaded: Paper[] = [];
    for (const file of pdfs) {
      try {
        const paper = await api<Paper>("/api/papers", {
          method: "POST",
          headers: { "Content-Type": "application/pdf", "X-Filename": file.name },
          body: file,
        });
        uploaded.push(paper);
        setPapers((current) => [paper, ...current.filter((item) => item.id !== paper.id)]);
        setSelectedId(paper.id);
      } catch (error) {
        setNotice({ kind: "error", text: error instanceof Error ? error.message : `Could not upload ${file.name}.` });
      } finally {
        setUploading((current) => Math.max(0, current - 1));
      }
    }
    if (uploaded.length) {
      setSelectedIds(uploaded.map((paper) => paper.id));
      setNotice({
        kind: "success",
        text: `${uploaded.length} paper${uploaded.length === 1 ? "" : "s"} extracted and ready.`,
      });
    }
    if (fileInput.current) fileInput.current.value = "";
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    void upload(Array.from(event.target.files ?? []));
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    void upload(Array.from(event.dataTransfer.files));
  }

  async function runStage(stage: RunnableStage, paperIds = selected ? [selected.id] : []) {
    if (!paperIds.length) return;
    try {
      const result = await api<{ jobs: Job[] }>("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paper_ids: paperIds, stages: [stage], options }),
      });
      setJobs((current) => [...result.jobs, ...current]);
      setTab("overview");
      setNotice({ kind: "success", text: `${stage} queued for ${paperIds.length} paper${paperIds.length === 1 ? "" : "s"}.` });
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : "Could not queue the stage." });
    }
  }

  async function runAll(paperIds = selected ? [selected.id] : []) {
    if (!paperIds.length) return;
    try {
      const created: Job[] = [];
      for (const paperId of paperIds) {
        const paper = papers.find((item) => item.id === paperId);
        if (!paper) continue;
        const stages = nextStages(paper);
        if (!stages.length) continue;
        const result = await api<{ jobs: Job[] }>("/api/jobs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ paper_ids: [paperId], stages, options }),
        });
        created.push(...result.jobs);
      }
      setJobs((current) => [...created, ...current]);
      setTab("overview");
      setNotice({ kind: "success", text: `${created.length} complete pipeline job${created.length === 1 ? "" : "s"} queued.` });
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : "Could not queue the pipeline." });
    }
  }

  async function cancelJob(jobId: string) {
    const job = await api<Job>(`/api/jobs/${jobId}`, { method: "DELETE" });
    setJobs((current) => current.map((item) => (item.id === job.id ? job : item)));
  }

  async function openArtifact(name: string) {
    if (!selected) return;
    setArtifactLoading(true);
    setArtifact({ name, content: "", dirty: false });
    try {
      const content = await textArtifact(selected.id, name);
      setArtifact({ name, content, dirty: false });
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : "Could not open artifact." });
    } finally {
      setArtifactLoading(false);
    }
  }

  async function activateTab(next: WorkspaceTab) {
    setTab(next);
    if (!selected) return;
    try {
      if (next === "topology" && !topology && selected.availableArtifacts.includes("topology")) {
        const content = await textArtifact(selected.id, "topology");
        setTopology(JSON.parse(content) as Topology);
      } else if (next === "evidence" && !evidence) {
        setEvidence(await api<Evidence>(`/api/papers/${encodeURIComponent(selected.id)}/evidence`));
      } else if (next === "code" && (!artifact || artifact.name !== "code")) {
        await openArtifact("code");
        const result = await api<{ revisions: Revision[] }>(
          `/api/papers/${encodeURIComponent(selected.id)}/revisions`,
        );
        setRevisions(result.revisions.filter((revision) => revision.artifact === "code"));
      } else if (next === "validation" && !validation) {
        const content = await textArtifact(selected.id, "validation");
        setValidation(JSON.parse(content) as ValidationDetail);
      }
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : `Could not open ${next}.` });
    }
  }

  async function saveTopology() {
    if (!selected || !topology) return;
    try {
      const paper = await api<Paper>(
        `/api/papers/${encodeURIComponent(selected.id)}/artifacts/topology`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(topology),
        },
      );
      setPapers((current) => current.map((item) => (item.id === paper.id ? paper : item)));
      setTopologyDirty(false);
      setNotice({ kind: "success", text: "Topology validated and saved. The previous version is retained." });
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : "Topology could not be saved." });
    }
  }

  async function saveCode() {
    if (!selected || !artifact || artifact.name !== "code") return;
    try {
      const paper = await api<Paper>(
        `/api/papers/${encodeURIComponent(selected.id)}/artifacts/code`,
        {
          method: "PUT",
          headers: { "Content-Type": "text/plain" },
          body: artifact.content,
        },
      );
      setPapers((current) => current.map((item) => (item.id === paper.id ? paper : item)));
      setArtifact({ ...artifact, dirty: false });
      const result = await api<{ revisions: Revision[] }>(
        `/api/papers/${encodeURIComponent(selected.id)}/revisions`,
      );
      setRevisions(result.revisions.filter((revision) => revision.artifact === "code"));
      setNotice({ kind: "success", text: "Code passed static validation and was saved." });
    } catch (error) {
      setNotice({ kind: "error", text: error instanceof Error ? error.message : "Code could not be saved." });
    }
  }

  function updateLayer(patch: Partial<LayerSpec>) {
    if (!topology || !selectedLayerId) return;
    setTopology({
      ...topology,
      layers: topology.layers.map((layer) => (
        layer.id === selectedLayerId ? { ...layer, ...patch } : layer
      )),
    });
    setTopologyDirty(true);
  }

  function addLayer() {
    if (!topology) return;
    let index = topology.layers.length + 1;
    let id = `layer_${index}`;
    while (topology.layers.some((layer) => layer.id === id)) {
      index += 1;
      id = `layer_${index}`;
    }
    setTopology({
      ...topology,
      layers: [
        ...topology.layers,
        {
          id,
          layer_type: "new_layer",
          inputs: [],
          parameters: {},
          confidence: 0.5,
          description: "User-added topology layer.",
        },
      ],
    });
    setSelectedLayerId(id);
    setTopologyDirty(true);
  }

  function removeLayer() {
    if (!topology || !selectedLayerId) return;
    setTopology({
      ...topology,
      layers: topology.layers.filter((layer) => layer.id !== selectedLayerId),
      connections: topology.connections.filter(
        (connection) => connection.source !== selectedLayerId && connection.target !== selectedLayerId,
      ),
    });
    setSelectedLayerId(null);
    setTopologyDirty(true);
  }

  async function updatePaper(patch: Partial<Pick<Paper, "title" | "tags" | "archived">>) {
    if (!selected) return;
    const paper = await api<Paper>(`/api/papers/${encodeURIComponent(selected.id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
    setPapers((current) => current.map((item) => (item.id === paper.id ? paper : item)));
  }

  async function duplicatePaper() {
    if (!selected) return;
    const paper = await api<Paper>(`/api/papers/${encodeURIComponent(selected.id)}/duplicate`, {
      method: "POST",
    });
    setPapers((current) => [paper, ...current]);
    setSelectedId(paper.id);
    setNotice({ kind: "success", text: "A separate run was created from this paper." });
  }

  async function deletePaper() {
    if (!selected || !window.confirm(`Move “${selected.title}” to TorchForge trash?`)) return;
    await api(`/api/papers/${encodeURIComponent(selected.id)}`, { method: "DELETE" });
    setPapers((current) => current.filter((paper) => paper.id !== selected.id));
    setSelectedId(papers.find((paper) => paper.id !== selected.id)?.id ?? null);
    setNotice({ kind: "success", text: "Paper moved to the recoverable project trash." });
  }

  async function loadRevision(revision: Revision) {
    if (!selected) return;
    const name = revision.path.split("/").at(-1);
    if (!name) return;
    const response = await fetch(
      `${apiBase()}/api/papers/${encodeURIComponent(selected.id)}/revisions/${encodeURIComponent(name)}`,
    );
    setSelectedRevision({ path: revision.path, content: await response.text() });
  }

  async function loadComparison(id: string) {
    setCompareId(id);
    if (!id) {
      setCompareCode("");
      return;
    }
    try {
      setCompareCode(await textArtifact(id, "code"));
    } catch {
      setCompareCode("");
    }
  }

  function download(path: string) {
    window.open(`${apiBase()}${path}`, "_blank", "noopener,noreferrer");
  }

  function addTag() {
    if (!selected || !tagDraft.trim()) return;
    void updatePaper({ tags: [...(selected.tags ?? []), tagDraft.trim()] });
    setTagDraft("");
  }

  const comparePaper = papers.find((paper) => paper.id === compareId) ?? null;

  return (
    <div className="studio-shell">
      <aside className={`studio-sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Zap size={18} /></div>
          <div><strong>TorchForge</strong><span>STUDIO 0.6</span></div>
          <button className="icon-button sidebar-close" onClick={() => setSidebarOpen(false)} aria-label="Close menu">
            <PanelLeftClose size={18} />
          </button>
        </div>

        <nav className="primary-nav" aria-label="Workspace navigation">
          <button className="active" onClick={() => { setTab("overview"); setSidebarOpen(false); }}>
            <Columns3 size={16} /> Workspace
          </button>
          <button onClick={() => { setTab("overview"); setSidebarOpen(false); }}>
            <Activity size={16} /> Pipeline queue <span>{activeJobs.length}</span>
          </button>
          <button onClick={() => document.getElementById("paper-library")?.scrollIntoView()}>
            <FileArchive size={16} /> Paper library <span>{papers.length}</span>
          </button>
        </nav>

        <section className="sidebar-section">
          <div className="sidebar-label">RECENT PAPERS</div>
          <div className="recent-list">
            {papers.slice(0, 6).map((paper) => (
              <button
                key={paper.id}
                className={paper.id === selectedId ? "selected" : ""}
                onClick={() => { setSelectedId(paper.id); setSidebarOpen(false); }}
              >
                <FileCode2 size={13} />
                <span>{paper.title}</span>
              </button>
            ))}
            {!papers.length && <p>No papers yet. Upload a PDF to begin.</p>}
          </div>
        </section>

        <div className="engine-card">
          <div><span className={`status-light ${engineOnline ? "online" : ""}`} /> Local engine</div>
          <strong>{engineOnline ? `${health?.device.toUpperCase()} ready` : "Offline"}</strong>
          <p>{health?.ollama.ready ? `${health.ollama.models.length} Ollama model(s) found` : "Ollama needs attention"}</p>
          <button onClick={() => void refresh()}><RefreshCw size={13} /> Refresh</button>
        </div>
      </aside>

      <main className="studio-main">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setSidebarOpen(true)} aria-label="Open menu">
            <Menu size={18} />
          </button>
          <div className="breadcrumbs"><span>Research workspace</span><ChevronDown size={12} /><strong>{selected?.title || "New paper"}</strong></div>
          <div className="topbar-actions">
            <div className={`connection-pill ${engineOnline ? "connected" : ""}`}>
              <span className={`status-light ${engineOnline ? "online" : ""}`} />
              {engineOnline ? "Engine connected" : "Engine offline"}
            </div>
            <button className="icon-button" onClick={() => setSettingsOpen(true)} aria-label="Open pipeline settings">
              <Settings2 size={17} />
            </button>
          </div>
        </header>

        <div className="studio-content">
          <section className="hero">
            <div>
              <span className="kicker"><Sparkles size={13} /> LOCAL RESEARCH ENGINEERING</span>
              <h1>Forge research into <span>working models.</span></h1>
              <p>Inspect evidence, correct architecture graphs, generate PyTorch, and prove runtime behavior from one auditable workspace.</p>
            </div>
            <div className="hero-stats">
              <div><strong>{papers.length}</strong><span>Papers</span></div>
              <div><strong>{jobs.filter((job) => job.status === "completed").length}</strong><span>Runs</span></div>
              <div><strong>{health?.device.toUpperCase() || "—"}</strong><span>Device</span></div>
            </div>
          </section>

          {!engineOnline && (
            <section className="guidance-banner">
              <div className="guidance-icon"><Server size={19} /></div>
              <div>
                <strong>Start the local TorchForge engine</strong>
                <p>Run <code>uv run torchforge serve</code>, then refresh this page.</p>
              </div>
              <button onClick={() => navigator.clipboard.writeText("uv run torchforge serve")}>
                <Clipboard size={14} /> Copy command
              </button>
            </section>
          )}

          {engineOnline && !health?.ollama.ready && (
            <section className="guidance-banner warning">
              <div className="guidance-icon"><AlertCircle size={19} /></div>
              <div>
                <strong>Ollama is not available</strong>
                <p>Start <code>ollama serve</code>, then pull <code>{options.vision_model}</code> and <code>{options.code_model}</code>.</p>
              </div>
              <button onClick={() => setSettingsOpen(true)}><Settings2 size={14} /> Configure</button>
            </section>
          )}

          <section
            className={`upload-command ${dragging ? "dragging" : ""}`}
            onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <div className="upload-orbit">{uploading ? <LoaderCircle className="spin" size={27} /> : <UploadCloud size={27} />}</div>
            <div>
              <strong>{uploading ? `Extracting ${uploading} paper${uploading === 1 ? "" : "s"}…` : "Drop a research paper here — or several"}</strong>
              <span>Multiple PDFs supported · up to 100 MB each</span>
            </div>
            <input ref={fileInput} type="file" accept="application/pdf,.pdf" multiple onChange={onFileChange} hidden />
            <button disabled={!engineOnline || uploading > 0} onClick={() => fileInput.current?.click()}>
              <Plus size={15} /> Choose PDFs
            </button>
          </section>

          {selected && (
            <section className="paper-workspace">
              <header className="paper-header">
                <div>
                  <span className="eyebrow">ACTIVE PAPER · {compactId(selected.id)}</span>
                  <div className="editable-title">
                    <h2>{selected.title}</h2>
                    <button
                      className="ghost-icon"
                      onClick={() => {
                        const title = window.prompt("Paper title", selected.title);
                        if (title?.trim()) void updatePaper({ title: title.trim() });
                      }}
                      aria-label="Rename paper"
                    ><Pencil size={14} /></button>
                  </div>
                  <p>{selected.source} · {selected.pageCount} pages · {stageProgress(selected)}/4 phases</p>
                </div>
                <div className="paper-actions">
                  <button className="secondary-button" onClick={() => void duplicatePaper()}><Copy size={14} /> New run</button>
                  <button className="secondary-button" onClick={() => void updatePaper({ archived: !selected.archived })}>
                    <Archive size={14} /> {selected.archived ? "Restore" : "Archive"}
                  </button>
                  <button className="danger-button" onClick={() => void deletePaper()}><Trash2 size={14} /> Delete</button>
                  <button
                    className="primary-button"
                    disabled={!nextStages(selected).length || !health?.ollama.ready}
                    onClick={() => void runAll()}
                  ><Play size={14} /> Run remaining</button>
                </div>
              </header>

              <div className="tag-row">
                <Tags size={13} />
                {(selected.tags ?? []).map((tag) => (
                  <button key={tag} onClick={() => void updatePaper({ tags: (selected.tags ?? []).filter((item) => item !== tag) })}>
                    {tag} <X size={11} />
                  </button>
                ))}
                <input
                  value={tagDraft}
                  onChange={(event) => setTagDraft(event.target.value)}
                  onKeyDown={(event) => { if (event.key === "Enter") addTag(); }}
                  placeholder="Add tag"
                />
              </div>

              <div className="pipeline-rail">
                {stageDefinitions.map((stage, index) => {
                  const done = selected.stages[stage.id];
                  const running = selectedJobs.some((job) => job.stage === stage.id && job.status === "running");
                  const priorDone = index === 0 || selected.stages[stageDefinitions[index - 1].id];
                  const Icon = stage.icon;
                  return (
                    <article key={stage.id} className={`${done ? "done" : ""} ${running ? "running" : ""}`}>
                      <div className="stage-index">{stage.number}</div>
                      <div className="stage-icon"><Icon size={17} /></div>
                      <div><strong>{stage.title}</strong><span>{running ? "Running now" : stage.description}</span></div>
                      {done ? (
                        <button
                          className="stage-state"
                          onClick={() => {
                            if (stage.id === "extract") void openArtifact("text");
                            else void activateTab(stage.id === "parse" ? "topology" : stage.id === "compile" ? "code" : "validation");
                          }}
                        ><Check size={13} /> Open</button>
                      ) : stage.id !== "extract" ? (
                        <button
                          className="stage-state"
                          disabled={!priorDone || running || !health?.ollama.ready}
                          onClick={() => void runStage(stage.id)}
                        ><Play size={12} /> Run</button>
                      ) : <span className="stage-state muted">Waiting</span>}
                    </article>
                  );
                })}
              </div>

              <div className="workspace-tabs" role="tablist">
                {tabs.map((item) => {
                  const Icon = item.icon;
                  return (
                    <button
                      key={item.id}
                      className={tab === item.id ? "active" : ""}
                      onClick={() => void activateTab(item.id)}
                      role="tab"
                      aria-selected={tab === item.id}
                    ><Icon size={14} /> {item.label}</button>
                  );
                })}
              </div>

              <div className="workspace-panel">
                {tab === "overview" && (
                  <div className="overview-grid">
                    <section className="panel-card queue-card">
                      <div className="panel-title"><div><span className="eyebrow">LIVE PIPELINE</span><h3>Jobs and stage logs</h3></div><Activity size={18} /></div>
                      {!selectedJobs.length && <div className="empty-state">No jobs for this paper yet. Run a stage to see live progress.</div>}
                      {selectedJobs.slice(0, 5).map((job) => (
                        <article className="job-row" key={job.id}>
                          <div className={`job-dot ${job.status}`} />
                          <div className="job-main">
                            <div><strong>{job.stage || job.stages.join(" → ")}</strong><span>{job.status}</span></div>
                            <div className="progress-track"><span style={{ width: `${job.progress}%` }} /></div>
                            <p>{job.logs.at(-1)?.message}</p>
                          </div>
                          <div className="job-meta">
                            <span>{job.progress}%</span>
                            <span>{formatDuration(job.durationMs)}</span>
                            {["queued", "running", "cancelling"].includes(job.status) && (
                              <button onClick={() => void cancelJob(job.id)} aria-label="Cancel job"><CircleStop size={15} /></button>
                            )}
                          </div>
                        </article>
                      ))}
                    </section>

                    <section className="panel-card">
                      <div className="panel-title"><div><span className="eyebrow">PROVENANCE</span><h3>Models and runtime</h3></div><Cpu size={18} /></div>
                      <dl className="metric-list">
                        <div><dt>Vision model</dt><dd>{selected.visionModel || options.vision_model}</dd></div>
                        <div><dt>Code model</dt><dd>{selected.codeModel || options.code_model}</dd></div>
                        <div><dt>Runtime device</dt><dd>{selected.validation?.device.toUpperCase() || health?.device.toUpperCase()}</dd></div>
                        <div><dt>Architecture</dt><dd>{selected.validation?.architecture_profile?.replaceAll("_", " ") || "Vision-derived"}</dd></div>
                        <div><dt>Repairs</dt><dd>{selected.validation?.attempt_count ?? 0}</dd></div>
                      </dl>
                    </section>

                    <section className="panel-card export-card">
                      <div className="panel-title"><div><span className="eyebrow">EXPORT CENTER</span><h3>Take the work with you</h3></div><Download size={18} /></div>
                      <div className="export-grid">
                        <button onClick={() => download(`/api/papers/${encodeURIComponent(selected.id)}/exports/bundle`)}>
                          <FileArchive size={18} /><span><strong>Artifact bundle</strong><small>Source, graph, code and reports</small></span>
                        </button>
                        <button onClick={() => download(`/api/papers/${encodeURIComponent(selected.id)}/exports/model-card`)}>
                          <FileDown size={18} /><span><strong>Model card</strong><small>Provenance and limitations</small></span>
                        </button>
                        <button
                          disabled={!selected.stages.validate}
                          onClick={() => download(`/api/papers/${encodeURIComponent(selected.id)}/exports/onnx`)}
                        >
                          <Boxes size={18} /><span><strong>ONNX model</strong><small>Portable runtime export</small></span>
                        </button>
                      </div>
                    </section>
                  </div>
                )}

                {tab === "topology" && (
                  <div className="topology-workbench">
                    {!topology ? (
                      <div className="empty-state large">
                        <Layers3 size={28} />
                        <strong>No topology available</strong>
                        <span>Run the Map phase to create an editable architecture graph.</span>
                      </div>
                    ) : (
                      <>
                        <div className="graph-toolbar">
                          <div><strong>{topology.architecture_name}</strong><span>{topology.layers.length} layers · {Math.round(topology.overall_confidence * 100)}% confidence</span></div>
                          <div>
                            <button className="secondary-button" onClick={addLayer}><Plus size={14} /> Layer</button>
                            <button className="primary-button" disabled={!topologyDirty} onClick={() => void saveTopology()}><Save size={14} /> Save graph</button>
                          </div>
                        </div>
                        <div className="graph-canvas">
                          <ReactFlow
                            nodes={graphNodes}
                            edges={graphEdges}
                            onNodesChange={(changes: NodeChange[]) => setGraphNodes((nodes) => applyNodeChanges(changes, nodes))}
                            onEdgesChange={(changes: EdgeChange[]) => setGraphEdges((edges) => applyEdgeChanges(changes, edges))}
                            onNodeClick={(_, node) => setSelectedLayerId(node.id)}
                            onConnect={(connection) => {
                              setGraphEdges((edges) => addEdge({ ...connection, markerEnd: { type: MarkerType.ArrowClosed } }, edges));
                              if (connection.source && connection.target) {
                                setTopology((current) => current ? {
                                  ...current,
                                  connections: [...current.connections, { source: connection.source!, target: connection.target!, kind: "sequential" }],
                                } : current);
                                setTopologyDirty(true);
                              }
                            }}
                            fitView
                          >
                            <Background gap={22} size={1} />
                            <MiniMap pannable zoomable />
                            <Controls />
                          </ReactFlow>
                        </div>
                        <aside className="layer-inspector">
                          {selectedLayer ? (
                            <>
                              <div className="panel-title"><div><span className="eyebrow">LAYER INSPECTOR</span><h3>{selectedLayer.id}</h3></div><button className="ghost-icon danger" onClick={removeLayer}><Trash2 size={15} /></button></div>
                              <label>Layer type<input value={selectedLayer.layer_type} onChange={(event) => updateLayer({ layer_type: event.target.value })} /></label>
                              <label>Confidence<input type="range" min="0" max="1" step="0.01" value={selectedLayer.confidence} onChange={(event) => updateLayer({ confidence: Number(event.target.value) })} /><span>{Math.round(selectedLayer.confidence * 100)}%</span></label>
                              <label>Description<textarea value={selectedLayer.description || ""} onChange={(event) => updateLayer({ description: event.target.value })} /></label>
                              <div className="shape-pair"><div><span>Input</span><strong>{formatShape(selectedLayer.input_shape)}</strong></div><div><span>Output</span><strong>{formatShape(selectedLayer.output_shape)}</strong></div></div>
                              <label>Parameters<textarea value={JSON.stringify(selectedLayer.parameters, null, 2)} onChange={(event) => { try { updateLayer({ parameters: JSON.parse(event.target.value) as Record<string, unknown> }); } catch { /* Keep the last valid object. */ } }} /></label>
                            </>
                          ) : (
                            <div className="empty-state">Select a node to inspect evidence, shapes, parameters, and confidence.</div>
                          )}
                        </aside>
                      </>
                    )}
                  </div>
                )}

                {tab === "evidence" && (
                  <div className="evidence-workbench">
                    <section className="pdf-panel">
                      <div className="panel-title"><div><span className="eyebrow">SOURCE PAPER</span><h3>{selected.source}</h3></div><FileText size={18} /></div>
                      {evidence?.sourceAvailable ? (
                        <iframe title={`Source PDF for ${selected.title}`} src={`${apiBase()}/api/papers/${encodeURIComponent(selected.id)}/source`} />
                      ) : (
                        <div className="empty-state large">The original PDF is outside the managed input directory and is not exposed by the API.</div>
                      )}
                    </section>
                    <aside className="evidence-strip">
                      <div className="panel-title"><div><span className="eyebrow">FIGURE EVIDENCE</span><h3>{evidence?.images.length || 0} candidates</h3></div><Layers3 size={18} /></div>
                      {selectedLayer && (
                        <div className="selected-evidence">
                          <span>Selected layer</span>
                          <strong>{selectedLayer.layer_type}</strong>
                          <p>{selectedLayer.description || "No layer-level evidence note was recorded."}</p>
                        </div>
                      )}
                      <div className="evidence-gallery">
                        {evidence?.images.map((image) => (
                          <figure key={image.path}>
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img src={evidenceUrl(selected.id, image.path)} alt={`Evidence from page ${image.page || "unknown"}`} />
                            <figcaption>Page {image.page || "—"} · {image.kind}</figcaption>
                          </figure>
                        ))}
                      </div>
                    </aside>
                  </div>
                )}

                {tab === "code" && (
                  <div className="code-workbench">
                    <div className="editor-toolbar">
                      <div><span className="eyebrow">GENERATED MODULE</span><strong>{selected.title}.py</strong></div>
                      <div>
                        <button className="secondary-button" onClick={() => navigator.clipboard.writeText(artifact?.content || "")}><Clipboard size={14} /> Copy</button>
                        <button className="secondary-button" onClick={() => download(`/api/papers/${encodeURIComponent(selected.id)}/artifacts/code`)}><Download size={14} /> Download</button>
                        <button className="primary-button" disabled={!artifact?.dirty} onClick={() => void saveCode()}><Save size={14} /> Save & check</button>
                      </div>
                    </div>
                    {artifactLoading ? (
                      <div className="empty-state large"><LoaderCircle className="spin" size={24} /> Loading code…</div>
                    ) : selectedRevision ? (
                      <DiffEditor
                        height="540px"
                        language="python"
                        original={selectedRevision.content}
                        modified={artifact?.content || ""}
                        theme="vs-dark"
                        options={{ readOnly: true, renderSideBySide: true, minimap: { enabled: false } }}
                      />
                    ) : (
                      <Editor
                        height="540px"
                        language="python"
                        value={artifact?.content || ""}
                        theme="vs-dark"
                        onChange={(value) => setArtifact((current) => current ? { ...current, content: value || "", dirty: true } : current)}
                        options={{ minimap: { enabled: false }, fontSize: 13, tabSize: 4, automaticLayout: true }}
                      />
                    )}
                    <div className="revision-bar">
                      <strong>Revision history</strong>
                      <button className={!selectedRevision ? "active" : ""} onClick={() => setSelectedRevision(null)}>Current</button>
                      {revisions.slice().reverse().map((revision) => (
                        <button key={revision.path} className={selectedRevision?.path === revision.path ? "active" : ""} onClick={() => void loadRevision(revision)}>
                          {new Date(revision.created_at).toLocaleString()}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {tab === "validation" && (
                  <div className="validation-dashboard">
                    {!validation ? (
                      <div className="empty-state large"><Gauge size={28} /><strong>No validation report</strong><span>Run validation to inspect shapes, checks, traces, and repairs.</span></div>
                    ) : (
                      <>
                        <div className="validation-summary">
                          <div><span>Status</span><strong className={validation.status === "failed" ? "failed" : "passed"}>{validation.status}</strong></div>
                          <div><span>Device</span><strong>{validation.device.toUpperCase()}</strong></div>
                          <div><span>Profile</span><strong>{validation.architecture_profile?.replaceAll("_", " ") || "Generic runtime"}</strong></div>
                          <div><span>Attempts</span><strong>{validation.attempts?.length || 0}</strong></div>
                        </div>
                        {validation.performance && (
                          <div className="validation-summary">
                            <div><span>P50 latency</span><strong>{validation.performance.latency_ms_p50.toFixed(2)} ms</strong></div>
                            <div><span>Mean / P95</span><strong>{validation.performance.latency_ms_mean.toFixed(2)} / {validation.performance.latency_ms_p95.toFixed(2)} ms</strong></div>
                            <div><span>Throughput</span><strong>{validation.performance.throughput_samples_per_sec.toFixed(1)} samples/s</strong></div>
                            <div><span>FLOPs (fwd)</span><strong>{formatFlops(validation.performance.estimated_flops)}</strong></div>
                            <div><span>Peak memory</span><strong>{formatBytes(validation.performance.peak_memory_bytes)}</strong></div>
                          </div>
                        )}
                        <div className="validation-columns">
                          <section className="panel-card">
                            <div className="panel-title"><div><span className="eyebrow">CONFORMANCE</span><h3>Architecture checks</h3></div><Check size={18} /></div>
                            <div className="check-list">
                              {validation.conformance_checks?.map((check) => (
                                <article key={check.name} className={check.passed ? "passed" : "failed"}>
                                  {check.passed ? <Check size={15} /> : <X size={15} />}
                                  <div><strong>{check.name}</strong><p>{check.detail}</p></div>
                                </article>
                              ))}
                              {!validation.conformance_checks?.length && <div className="empty-state">Generic runtime validation completed without a certified profile.</div>}
                            </div>
                          </section>
                          <section className="panel-card">
                            <div className="panel-title"><div><span className="eyebrow">RUNTIME CONTRACT</span><h3>Shapes and constructor</h3></div><Cpu size={18} /></div>
                            <dl className="metric-list">
                              <div><dt>Class</dt><dd>{validation.class_name}</dd></div>
                              <div><dt>Inputs</dt><dd>{validation.input_shapes?.map(formatShape).join(" · ")}</dd></div>
                              <div><dt>Outputs</dt><dd>{validation.output_shapes?.map(formatShape).join(" · ")}</dd></div>
                              <div><dt>Arguments</dt><dd><code>{JSON.stringify(validation.constructor_kwargs || {})}</code></dd></div>
                            </dl>
                          </section>
                        </div>
                        <section className="panel-card attempts-card">
                          <div className="panel-title"><div><span className="eyebrow">REPAIR HISTORY</span><h3>Validation attempts</h3></div><RefreshCw size={18} /></div>
                          {validation.attempts?.map((attempt) => (
                            <details key={attempt.attempt}>
                              <summary><span>Attempt {attempt.attempt}</span><strong className={attempt.succeeded ? "passed" : "failed"}>{attempt.succeeded ? "Passed" : "Failed"}</strong></summary>
                              <pre>{attempt.error || "Forward execution and conformance checks completed successfully."}</pre>
                            </details>
                          ))}
                        </section>
                      </>
                    )}
                  </div>
                )}

                {tab === "compare" && (
                  <div className="compare-workbench">
                    <div className="compare-toolbar">
                      <div><span className="eyebrow">RUN COMPARISON</span><h3>Compare generated modules and provenance</h3></div>
                      <select value={compareId} onChange={(event) => void loadComparison(event.target.value)}>
                        <option value="">Choose another paper or run</option>
                        {papers.filter((paper) => paper.id !== selected.id).map((paper) => (
                          <option key={paper.id} value={paper.id}>{paper.title} · {compactId(paper.id)}</option>
                        ))}
                      </select>
                    </div>
                    {comparePaper ? (
                      <>
                        <div className="comparison-metrics">
                          <div><span>Paper</span><strong>{selected.title}</strong><strong>{comparePaper.title}</strong></div>
                          <div><span>Vision</span><strong>{selected.visionModel || "—"}</strong><strong>{comparePaper.visionModel || "—"}</strong></div>
                          <div><span>Code model</span><strong>{selected.codeModel || "—"}</strong><strong>{comparePaper.codeModel || "—"}</strong></div>
                          <div><span>Validation</span><strong>{selected.validation?.status || "—"}</strong><strong>{comparePaper.validation?.status || "—"}</strong></div>
                          <div><span>Outputs</span><strong>{selected.validation?.output_shapes?.map(formatShape).join(" · ") || "—"}</strong><strong>{comparePaper.validation?.output_shapes?.map(formatShape).join(" · ") || "—"}</strong></div>
                        </div>
                        <DiffEditor
                          height="520px"
                          language="python"
                          original={compareCode}
                          modified={artifact?.name === "code" ? artifact.content : ""}
                          theme="vs-dark"
                          options={{ readOnly: true, minimap: { enabled: false }, renderSideBySide: true }}
                          beforeMount={() => { if (!artifact || artifact.name !== "code") void openArtifact("code"); }}
                        />
                      </>
                    ) : <div className="empty-state large"><GitCompareArrows size={28} /><strong>Select another run</strong><span>Compare models, validation results, shapes, and generated source side by side.</span></div>}
                  </div>
                )}
              </div>
            </section>
          )}

          <section className="paper-library" id="paper-library">
            <header>
              <div><span className="eyebrow">LOCAL ARTIFACT STORE</span><h2>Paper library</h2></div>
              <div className="library-tools">
                <label className="search-field"><Search size={14} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search papers, IDs or tags" /></label>
                <label className="archive-toggle"><input type="checkbox" checked={showArchived} onChange={(event) => setShowArchived(event.target.checked)} /> Archived</label>
                {selectedIds.length > 0 && (
                  <button className="primary-button" onClick={() => void runAll(selectedIds)}>
                    <Play size={13} /> Run selected ({selectedIds.length})
                  </button>
                )}
              </div>
            </header>
            {loading ? (
              <div className="empty-state large"><LoaderCircle className="spin" size={23} /> Loading local papers…</div>
            ) : filtered.length ? (
              <div className="library-table">
                <div className="library-row table-head"><span /><span>Paper</span><span>Progress</span><span>Models</span><span>Status</span></div>
                {filtered.map((paper) => (
                  <div className={`library-row ${paper.id === selectedId ? "selected" : ""}`} key={paper.id}>
                    <input
                      type="checkbox"
                      checked={selectedIds.includes(paper.id)}
                      onChange={(event) => setSelectedIds((current) => event.target.checked ? [...current, paper.id] : current.filter((id) => id !== paper.id))}
                      aria-label={`Select ${paper.title}`}
                    />
                    <button onClick={() => { setSelectedId(paper.id); window.scrollTo({ top: 500, behavior: "smooth" }); }}>
                      <strong>{paper.title}</strong><span>{paper.source} · {compactId(paper.id)}</span>
                    </button>
                    <div><div className="mini-progress"><span style={{ width: `${stageProgress(paper) * 25}%` }} /></div><small>{stageProgress(paper)}/4 phases</small></div>
                    <div><strong>{paper.visionModel || "—"}</strong><span>{paper.codeModel || "—"}</span></div>
                    <div><span className={`status-badge ${paper.validation?.status || paper.status}`}>{paper.validation?.status || paper.status}</span></div>
                  </div>
                ))}
              </div>
            ) : <div className="empty-state large">No papers match the current filters.</div>}
          </section>
        </div>
      </main>

      {settingsOpen && (
        <div className="drawer-layer">
          <button className="drawer-scrim" onClick={() => setSettingsOpen(false)} aria-label="Close settings" />
          <aside className="settings-drawer">
            <header><div><span className="eyebrow">PIPELINE CONTROL</span><h2>Run settings</h2></div><button className="icon-button" onClick={() => setSettingsOpen(false)}><X size={18} /></button></header>
            <div className="preset-row">
              <button onClick={() => setOptions({ ...defaultOptions, max_images: 4, context_window: 4096, max_output_tokens: 2048, max_repairs: 1 })}>Fast local</button>
              <button onClick={() => setOptions(defaultOptions)}>Balanced</button>
              <button onClick={() => setOptions({ ...defaultOptions, max_images: 16, context_window: 16384, max_output_tokens: 8192, max_repairs: 3 })}>Thorough</button>
            </div>
            <label>Vision model<input list="ollama-models" value={options.vision_model} onChange={(event) => setOptions({ ...options, vision_model: event.target.value })} /></label>
            <label>Code model<input list="ollama-models" value={options.code_model} onChange={(event) => setOptions({ ...options, code_model: event.target.value })} /></label>
            <datalist id="ollama-models">{health?.ollama.models.map((model) => <option value={model} key={model} />)}</datalist>
            <label>Validation device<select value={options.device} onChange={(event) => setOptions({ ...options, device: event.target.value as StageOptions["device"] })}><option value="auto">Automatic</option><option value="cpu">CPU</option><option value="mps">Apple MPS</option><option value="cuda">NVIDIA CUDA</option></select></label>
            <div className="settings-grid">
              <label>Maximum images<input type="number" min="1" max="32" value={options.max_images} onChange={(event) => setOptions({ ...options, max_images: Number(event.target.value) })} /></label>
              <label>Repair attempts<input type="number" min="0" max="5" value={options.max_repairs} onChange={(event) => setOptions({ ...options, max_repairs: Number(event.target.value) })} /></label>
              <label>Context window<input type="number" min="512" value={options.context_window} onChange={(event) => setOptions({ ...options, context_window: Number(event.target.value) })} /></label>
              <label>Output tokens<input type="number" min="256" value={options.max_output_tokens} onChange={(event) => setOptions({ ...options, max_output_tokens: Number(event.target.value) })} /></label>
              <label>Paper text chars<input type="number" min="500" value={options.max_text_chars} onChange={(event) => setOptions({ ...options, max_text_chars: Number(event.target.value) })} /></label>
              <label>Timeout seconds<input type="number" min="5" value={options.timeout} onChange={(event) => setOptions({ ...options, timeout: Number(event.target.value) })} /></label>
            </div>
            <section className="setup-guide">
              <strong>Environment checklist</strong>
              <p className={engineOnline ? "good" : ""}>{engineOnline ? "✓" : "1."} TorchForge API</p>
              <p className={health?.ollama.ready ? "good" : ""}>{health?.ollama.ready ? "✓" : "2."} Ollama service</p>
              <p className={health?.ollama.models.includes(options.vision_model) ? "good" : ""}>3. Vision model: <code>ollama pull {options.vision_model}</code></p>
              <p className={health?.ollama.models.includes(options.code_model) ? "good" : ""}>4. Code model: <code>ollama pull {options.code_model}</code></p>
            </section>
            <button className="primary-button drawer-save" onClick={() => setSettingsOpen(false)}><Check size={14} /> Save settings</button>
          </aside>
        </div>
      )}

      {notice && (
        <div className={`toast ${notice.kind}`}>
          {notice.kind === "success" ? <Check size={18} /> : <AlertCircle size={18} />}
          <span>{notice.text}</span>
          <button onClick={() => setNotice(null)} aria-label="Dismiss notification"><X size={15} /></button>
        </div>
      )}
    </div>
  );
}
