import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  BookOpen,
  ExternalLink,
  FileText,
  Info,
  LifeBuoy,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Wifi,
  Zap,
} from 'lucide-react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import './index.css';

type Prediction = {
  attack_detected: boolean;
  attack_probability: number;
  confidence: 'high' | 'medium' | 'low' | 'unknown';
  predicted_stage: string;
  trend: 'rising' | 'stable' | 'falling';
};

type Forecast = {
  attack_probability_timeline: number[];
  trend: 'rising' | 'stable' | 'falling';
  predicted_stage: string;
  stage_confidence: 'confident' | 'uncertain';
  top_features: { feature: string; contribution: number }[];
  explanation_text: string;
};

type SecurityZone = {
  zone: 'green' | 'yellow' | 'red';
  attack_probability: number;
  reason: string;
};

type Alert = {
  id: number;
  severity: string;
  message: string;
  source: string;
  resolved: boolean;
  created_at: number;
};

type Explain = {
  top_features: { feature: string; contribution: number }[];
  explanation_text: string;
};

type KnowledgeItem = {
  title: string;
  description: string;
  indicators: string[];
};

type HelpResource = {
  name: string;
  purpose: string;
  url: string;
  phone: string | null;
};

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}

async function postJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}

function pct(value: number | undefined) {
  if (value == null || Number.isNaN(value)) return '—';
  return `${Math.round(value * 100)}%`;
}

function zoneClass(zone: string) {
  if (zone === 'red') return 'border-red-200 bg-red-50 text-red-700';
  if (zone === 'yellow') return 'border-amber-200 bg-amber-50 text-amber-700';
  return 'border-emerald-200 bg-emerald-50 text-emerald-700';
}

function severityClass(severity: string) {
  if (/critical|high/i.test(severity)) return 'border-red-200 bg-red-50 text-red-700';
  if (/medium/i.test(severity)) return 'border-amber-200 bg-amber-50 text-amber-700';
  return 'border-slate-200 bg-slate-50 text-slate-600';
}

export default function App() {
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [forecast, setForecast] = useState<Forecast | null>(null);
  const [securityZone, setSecurityZone] = useState<SecurityZone | null>(null);
  const [explain, setExplain] = useState<Explain | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [knowledge, setKnowledge] = useState<KnowledgeItem[]>([]);
  const [helpResources, setHelpResources] = useState<HelpResource[]>([]);
  const [apiError, setApiError] = useState('');
  const [loading, setLoading] = useState(true);
  const [injecting, setInjecting] = useState(false);
  const [lastSync, setLastSync] = useState('—');
  const loadInFlight = useRef<Promise<void> | null>(null);

  const load = useCallback(() => {
    // Render Free can take a while to wake after inactivity. Do not let the
    // 10-second polling interval create overlapping requests while a previous
    // load is still waking the backend up.
    if (loadInFlight.current) return loadInFlight.current;

    const run = (async () => {
      try {
        // /predict, /security-zone and /explain depend on /rollout creating
        // the latest forecast. Treat /rollout as the wake-up/readiness request
        // so we do not fire six more requests while Render is still asleep.
        let forecast: Forecast;
        try {
          forecast = await getJson<Forecast>('/rollout?k=8');
        } catch {
          setApiError('Backend is waking up. Retrying automatically…');
          return;
        }

        setForecast(forecast);

        const dependentResults = await Promise.allSettled([
          getJson<Prediction>('/predict'),
          getJson<SecurityZone>('/security-zone'),
          getJson<Alert[]>('/alerts'),
          getJson<KnowledgeItem[]>('/knowledge-center'),
          getJson<HelpResource[]>('/help-resources'),
        ]);

        const [
          predictionResult,
          zoneResult,
          alertsResult,
          knowledgeResult,
          helpResult,
        ] = dependentResults;

        if (predictionResult.status === 'fulfilled') setPrediction(predictionResult.value);
        if (zoneResult.status === 'fulfilled') setSecurityZone(zoneResult.value);
        if (alertsResult.status === 'fulfilled') setAlerts(alertsResult.value);
        if (knowledgeResult.status === 'fulfilled') setKnowledge(knowledgeResult.value);
        if (helpResult.status === 'fulfilled') setHelpResources(helpResult.value);

        const failed = dependentResults.filter((result) => result.status === 'rejected').length;
        if (failed) {
          setApiError(
            `${failed} backend request${failed > 1 ? 's are' : ' is'} temporarily unavailable. Retrying automatically…`,
          );
        } else {
          setApiError('');
        }

        setLastSync(
          new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
          }),
        );
        setLoading(false);
      } finally {
        // Only clear the promise if it is still this load. This prevents an
        // older completion from unlocking a newer load accidentally.
        if (loadInFlight.current === run) {
          loadInFlight.current = null;
        }
      }
    })();

    loadInFlight.current = run;
    return run;
  }, []);

  useEffect(() => {
    void load();
    // The /rollout endpoint runs the model and SHAP explanation. Ten-second
    // polling was unnecessarily aggressive for a static demo state and can
    // increase memory pressure on small backend instances. Manual Refresh and
    // Inject Attack still trigger an immediate load.
    const timer = window.setInterval(() => void load(), 60000);
    return () => window.clearInterval(timer);
  }, [load]);

  const inject = async () => {
    setInjecting(true);
    try {
      await postJson('/alerts/trigger-attack');
      await load();
    } catch (error) {
      setApiError(
        `Inject Attack failed: ${error instanceof Error ? error.message : 'backend error'}`,
      );
    } finally {
      setInjecting(false);
    }
  };

  const currentAttack = Boolean(prediction?.attack_detected);
  const timeline = useMemo(
    () =>
      (forecast?.attack_probability_timeline ?? []).map((probability, index) => ({
        step: index + 1,
        probability: probability * 100,
      })),
    [forecast],
  );
  const activeAlerts = useMemo(
    () => alerts.filter((alert) => !alert.resolved),
    [alerts],
  );
  const features = forecast?.top_features ?? [];

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b bg-card/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-5 py-4 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-xl bg-primary text-primary-foreground">
              <ShieldCheck className="h-5 w-5" />
            </div>
            <div>
              <div className="font-display text-xl font-semibold">NetGuard AI</div>
              <div className="text-[10px] font-bold uppercase tracking-[.2em] text-muted-foreground">
                Predictive Cyber Defence Application
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="hidden text-xs text-muted-foreground sm:block">
              Last sync {lastSync}
            </span>
            <button
              onClick={() => void load()}
              className="rounded-xl border px-3 py-2 text-sm font-semibold hover:bg-muted"
            >
              <RefreshCw className="mr-2 inline h-4 w-4" />
              Refresh
            </button>
            <button
              onClick={() => void inject()}
              disabled={injecting}
              className="rounded-xl bg-primary px-4 py-2 text-sm font-bold text-primary-foreground disabled:opacity-60"
              title="Demo-only alert workflow; does not modify the network"
            >
              <Zap className="mr-2 inline h-4 w-4" />
              {injecting ? 'Injecting…' : 'Inject Attack'}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-5 px-5 py-6 lg:px-8">
        {apiError && (
          <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
            <AlertTriangle className="mr-2 inline h-4 w-4" />
            {apiError}
          </div>
        )}

        <section className="grid gap-5 lg:grid-cols-[1.1fr_.9fr]">
          <div
            className={`rounded-2xl border p-6 ${
              currentAttack
                ? 'border-red-200 bg-red-50'
                : 'border-emerald-200 bg-emerald-50'
            }`}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xs font-bold uppercase tracking-[.18em]">
                  Current detection
                </p>
                <h1 className="mt-2 font-display text-3xl font-semibold">
                  {loading && !prediction
                    ? 'Loading…'
                    : currentAttack
                      ? 'Attack detected'
                      : 'No active attack detected'}
                </h1>
                <p className="mt-2 max-w-xl text-sm text-muted-foreground">
                  Current detection is shown separately from the future
                  forecast returned by the backend.
                </p>
              </div>
              <div
                className={`grid h-14 w-14 place-items-center rounded-2xl ${
                  currentAttack
                    ? 'bg-red-100 text-red-700'
                    : 'bg-emerald-100 text-emerald-700'
                }`}
              >
                {currentAttack ? <ShieldAlert /> : <ShieldCheck />}
              </div>
            </div>

            <div className="mt-6 grid gap-3 sm:grid-cols-3">
              <Metric
                label="Detection probability"
                value={pct(prediction?.attack_probability)}
              />
              <Metric label="Confidence" value={prediction?.confidence ?? '—'} />
              <Metric
                label="Predicted stage"
                value={prediction?.predicted_stage ?? '—'}
              />
            </div>
          </div>

          <div className="rounded-2xl border bg-card p-6">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs font-bold uppercase tracking-[.18em] text-muted-foreground">
                  Forecast risk
                </p>
                <h2 className="mt-2 font-display text-2xl font-semibold">
                  {securityZone?.zone?.toUpperCase() ?? '—'}
                </h2>
              </div>
              {securityZone && (
                <span
                  className={`rounded-full border px-3 py-1 text-xs font-bold ${zoneClass(
                    securityZone.zone,
                  )}`}
                >
                  {pct(securityZone.attack_probability)}
                </span>
              )}
            </div>

            <div className="mt-6 grid grid-cols-2 gap-3">
              <Metric
                label="Forecast trend"
                value={forecast?.trend ?? prediction?.trend ?? '—'}
              />
              <Metric
                label="Stage confidence"
                value={forecast?.stage_confidence ?? '—'}
              />
            </div>

            <p className="mt-4 rounded-xl border bg-muted/30 p-3 text-sm text-muted-foreground">
              {securityZone?.reason ??
                'Risk explanation is not currently supplied by the backend.'}
            </p>
          </div>
        </section>

        <section className="grid gap-5 lg:grid-cols-2">
          <ChartCard
            title="Forecast probability timeline"
            subtitle="K-step values returned by the backend /rollout endpoint"
          >
            <div className="h-64">
              {timeline.length ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={timeline}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="step" />
                    <YAxis
                      domain={[0, 100]}
                      tickFormatter={(value) => `${value}%`}
                    />
                    <Tooltip
                      formatter={(value: number) => [
                        `${value.toFixed(1)}%`,
                        'Probability',
                      ]}
                    />
                    <ReferenceLine y={50} strokeDasharray="5 5" />
                    <Line
                      type="monotone"
                      dataKey="probability"
                      stroke="currentColor"
                      strokeWidth={3}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <Unavailable text="No forecast timeline was supplied by the connected backend." />
              )}
            </div>
          </ChartCard>

          <ChartCard
            title="Explainable AI"
            subtitle="Top contributing features supplied by M4"
          >
            <p className="text-sm leading-6 text-muted-foreground">
              {forecast?.explanation_text ??
                'No explanation was supplied by the backend.'}
            </p>
            <div className="mt-5 space-y-2">
              {features.length ? (
                features.slice(0, 5).map((feature) => (
                  <div
                    key={feature.feature}
                    className="flex items-center justify-between rounded-lg border bg-background px-3 py-2 text-sm"
                  >
                    <span>{feature.feature}</span>
                    <span className="font-semibold">
                      {feature.contribution.toFixed(2)}
                    </span>
                  </div>
                ))
              ) : (
                <Unavailable text="No feature contributions supplied." />
              )}
            </div>
          </ChartCard>
        </section>

        <section className="grid gap-5 lg:grid-cols-[1.1fr_.9fr]">
          <div className="rounded-2xl border bg-card p-6">
            <div className="flex items-center gap-3">
              <Activity className="h-5 w-5 text-primary" />
              <div>
                <h2 className="font-display text-xl font-semibold">
                  Model context
                </h2>
                <p className="text-xs text-muted-foreground">
                  Verified upstream information; not relabelled as classification metrics.
                </p>
              </div>
            </div>
            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              <Metric label="World model" value="GRU" />
              <Metric label="State dimension" value="78" />
              <Metric label="Sequence length" value="20" />
              <Metric label="Forecast horizon" value={`${timeline.length || 8} steps`} />
            </div>
          </div>

          <div className="rounded-2xl border bg-card p-6">
            <div className="flex items-center gap-3">
              <Wifi className="h-5 w-5 text-primary" />
              <div>
                <h2 className="font-display text-xl font-semibold">
                  Backend connection
                </h2>
                <p className="text-xs text-muted-foreground">
                  REST endpoints from the final M5/M7 contract
                </p>
              </div>
            </div>
            <div className="mt-5 space-y-3">
              <Status label="API base" value={API_BASE} />
              <Status label="Polling" value="Every 60 seconds" />
              <Status label="Forecast" value="/rollout?k=8" />
              <Status label="Live traffic" value="Not exposed by final backend" />
            </div>
          </div>
        </section>

        <section className="grid gap-5 lg:grid-cols-[1.1fr_.9fr]">
          <div className="rounded-2xl border bg-card p-6">
            <div className="flex items-end justify-between gap-3">
              <div>
                <p className="text-xs font-bold uppercase tracking-[.18em] text-muted-foreground">
                  Alerts
                </p>
                <h2 className="mt-2 font-display text-xl font-semibold">
                  Latest security alerts
                </h2>
              </div>
              <span className="rounded-full border px-3 py-1 text-xs font-bold">
                {activeAlerts.length} active
              </span>
            </div>

            <div className="mt-5 divide-y">
              {loading && !alerts.length ? (
                <div className="py-8 text-center text-sm text-muted-foreground">
                  Loading alerts…
                </div>
              ) : activeAlerts.length ? (
                activeAlerts.slice(0, 8).map((alert) => (
                  <div key={alert.id} className="flex gap-3 py-4 first:pt-0">
                    <div
                      className={`mt-1 grid h-9 w-9 shrink-0 place-items-center rounded-lg border ${severityClass(
                        alert.severity,
                      )}`}
                    >
                      <AlertTriangle className="h-4 w-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <p className="font-semibold">{alert.message}</p>
                        <span className="rounded-full border px-2 py-1 text-[10px] font-bold uppercase">
                          {alert.severity}
                        </span>
                      </div>
                      <p className="mt-1 text-sm text-muted-foreground">
                        {alert.source} ·{' '}
                        {new Date(alert.created_at * 1000).toLocaleString()}
                      </p>
                    </div>
                  </div>
                ))
              ) : (
                <Unavailable text="No active alerts were supplied." />
              )}
            </div>
          </div>

          <div className="rounded-2xl border bg-card p-6">
            <div className="flex items-center gap-3">
              <BookOpen className="h-5 w-5 text-primary" />
              <div>
                <h2 className="font-display text-xl font-semibold">
                  Knowledge Center
                </h2>
                <p className="text-xs text-muted-foreground">
                  M7 cybersecurity guidance supplied by the backend
                </p>
              </div>
            </div>

            <div className="mt-5 space-y-3">
              {knowledge.length ? (
                knowledge.map((item) => (
                  <div key={item.title} className="rounded-xl border p-4">
                    <p className="font-semibold">{item.title}</p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {item.description}
                    </p>
                    {item.indicators.length ? (
                      <p className="mt-2 text-xs text-muted-foreground">
                        Indicators: {item.indicators.join(' · ')}
                      </p>
                    ) : null}
                  </div>
                ))
              ) : (
                <Unavailable text="No knowledge-center content supplied." />
              )}
            </div>
          </div>
        </section>

        <section className="rounded-2xl border bg-card p-6">
          <div className="flex items-center gap-3">
            <LifeBuoy className="h-5 w-5 text-primary" />
            <div>
              <h2 className="font-display text-xl font-semibold">
                Cyber Safety & Assistance
              </h2>
              <p className="text-xs text-muted-foreground">
                M7 help resources returned by the backend
              </p>
            </div>
          </div>

          <div className="mt-5 grid gap-3 md:grid-cols-2">
            {helpResources.length ? (
              helpResources.map((resource) => (
                <div key={resource.name} className="rounded-xl border p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="font-semibold">{resource.name}</p>
                      <p className="mt-1 text-sm text-muted-foreground">
                        {resource.purpose}
                      </p>
                    </div>
                    <ExternalLink className="h-4 w-4 shrink-0 text-muted-foreground" />
                  </div>
                  <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
                    <a
                      href={resource.url}
                      target="_blank"
                      rel="noreferrer"
                      className="font-semibold text-primary underline-offset-4 hover:underline"
                    >
                      Open resource
                    </a>
                    {resource.phone ? (
                      <span className="rounded-full border px-2 py-1 text-xs font-semibold">
                        {resource.phone}
                      </span>
                    ) : null}
                  </div>
                </div>
              ))
            ) : (
              <Unavailable text="No help resources supplied." />
            )}
          </div>

          <div className="mt-5 flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
            <FileText className="mt-0.5 h-4 w-4 shrink-0" />
            <p>
              NetGuard provides information and local evidence generation; it
              does not automatically submit reports or take network actions.
            </p>
          </div>
        </section>

        <footer className="pb-4 text-center text-xs text-muted-foreground">
          NetGuard AI · M6 Application consuming the final M5/M7 REST contract
        </footer>
      </main>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border bg-background p-3">
      <div className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 truncate font-semibold">{value}</div>
    </div>
  );
}

function Status({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-xl border bg-background px-3 py-2 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-semibold">{value}</span>
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border bg-card p-6">
      <h2 className="font-display text-xl font-semibold">{title}</h2>
      <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>
      <div className="mt-4">{children}</div>
    </div>
  );
}

function Unavailable({ text }: { text: string }) {
  return (
    <div className="grid min-h-32 place-items-center rounded-xl border border-dashed bg-muted/20 p-6 text-center text-sm text-muted-foreground">
      {text}
    </div>
  );
}
