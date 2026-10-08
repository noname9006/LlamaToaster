import { useEffect, useState } from "react";
import { Routes, Route, useLocation, useNavigate } from "react-router-dom";
import { NavList, Sidebar } from "./components/Sidebar";
import { StatusBar } from "./components/StatusBar";
import { LtIcon } from "./components/ltIcons";
import { useWorkerStatuses } from "./api/useWorkerStatus";
import { ChatPanel } from "./components/ChatPanel";
import { Dashboard } from "./pages/Dashboard";
import { Models } from "./pages/Models";
import { Benchmark } from "./pages/Benchmark";
import { CustomTest } from "./pages/CustomTest";
import { Tests } from "./pages/Tests";
import { TestDetail } from "./pages/TestDetail";
import { Compare } from "./pages/Compare";
import { Workers } from "./pages/Workers";
import { Login } from "./pages/Login";
import { Settings } from "./pages/Settings";
import { Device } from "./pages/Device";
import { api } from "./api/client";
import type { AuthStatus } from "./types";

export default function App() {
  const location = useLocation();
  const navigate = useNavigate();
  // null = boot check not resolved yet. A failed fetch (network hiccup, not
  // an auth rejection -- getAuthStatus is itself a PUBLIC_PATH that never
  // 401s) is treated as "auth off" rather than blocking the whole app.
  const [authStatus, setAuthStatus] = useState<AuthStatus | null>(null);

  useEffect(() => {
    api
      .getAuthStatus()
      .then(setAuthStatus)
      .catch(() =>
        setAuthStatus({
          user: null,
          authEnabled: false,
          appSettings: {
            communitySharingAllowed: false,
            communityUserChoiceAllowed: true,
            accountDeletionAllowed: true,
            workerVramDiscrepancyPolicy: "warn",
            probeMaxLoads: 24,
          },
        })
      );
  }, []);

  const needsLogin = authStatus !== null && authStatus.authEnabled && authStatus.user === null;

  // MULTIUSER_PLAN.md §2.8: an unauthenticated visitor hitting any route
  // gets redirected to /login. Runs as an effect (not during render) since
  // it's a navigation, and `replace` so the protected route they originally
  // hit isn't left sitting in back-button history under a login wall.
  useEffect(() => {
    if (needsLogin && location.pathname !== "/login") {
      navigate("/login", { replace: true });
    }
  }, [needsLogin, location.pathname, navigate]);

  if (authStatus === null) {
    return <div className="flex min-h-screen items-center justify-center bg-bg text-sm text-muted">Loading…</div>;
  }

  // /login (and the moment right before the redirect effect above fires) is
  // a standalone full-screen page -- no Sidebar/ChatPanel chrome, since
  // there's no authenticated app to navigate yet.
  if (location.pathname === "/login" || needsLogin) {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="*" element={<Login />} />
      </Routes>
    );
  }

  return <Shell authStatus={authStatus} />;
}

// Narrow layouts (< 760px) swap the sidebar for a top bar with a menu drawer,
// as in the v2 design.
function useNarrow(): boolean {
  const query = "(max-width: 759px)";
  const [narrow, setNarrow] = useState(() => typeof window !== "undefined" && window.matchMedia?.(query).matches === true);
  useEffect(() => {
    const mq = window.matchMedia?.(query);
    if (!mq) return;
    const onChange = () => setNarrow(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return narrow;
}

function Shell({ authStatus }: { authStatus: AuthStatus }) {
  const narrow = useNarrow();
  const [drawer, setDrawer] = useState(false);
  const location = useLocation();
  const { workers } = useWorkerStatuses();
  useEffect(() => setDrawer(false), [location.pathname]);
  const footer = authStatus.user?.displayName ?? null;

  return (
    <div className="flex min-h-screen flex-wrap bg-bg">
      <a
        href="#lt-main"
        className="absolute -top-12 left-2 z-20 bg-accent px-3.5 py-2.5 font-semibold text-accent-fg focus:top-2"
      >
        Skip to content
      </a>
      {!narrow && <Sidebar authEnabled={authStatus.authEnabled} footer={footer} />}
      {narrow && (
        <>
          <header className="sticky top-0 z-[8] flex flex-[1_0_100%] items-center gap-2.5 border-b border-border bg-surface px-4 py-1.5">
            <img src="/toaster_favicon.png" alt="" className="h-7 w-7 object-contain" />
            <span className="font-display text-lg font-semibold">LlamaToaster</span>
            <button
              type="button"
              onClick={() => setDrawer((d) => !d)}
              aria-expanded={drawer}
              aria-controls="lt-drawer"
              className="ml-auto inline-flex min-h-11 items-center gap-2 border border-border-strong bg-transparent px-3 text-sm text-fg"
            >
              <LtIcon name="menu" />
              Menu
            </button>
          </header>
          {drawer && (
            <nav id="lt-drawer" aria-label="Main" className="flex flex-[1_0_100%] flex-col gap-0.5 border-b border-border bg-surface p-2">
              <NavList authEnabled={authStatus.authEnabled} large onNavigate={() => setDrawer(false)} />
            </nav>
          )}
        </>
      )}
      <main id="lt-main" className={`min-w-0 flex-[1_1_320px] ${narrow ? "" : "h-screen overflow-y-auto"}`}>
        <StatusBar workers={workers} sticky={!narrow} />
        <div className="w-full px-[clamp(16px,4vw,32px)] pb-10 pt-7">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/models" element={<Models />} />
            <Route path="/benchmark" element={<Benchmark />} />
            <Route path="/new" element={<Benchmark />} />
            <Route path="/custom-test" element={<CustomTest />} />
            <Route path="/tests" element={<Tests />} />
            <Route path="/tests/:id" element={<TestDetail />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/workers" element={<Workers />} />
            <Route path="/device" element={<Device />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </div>
      </main>
      <ChatPanel narrow={narrow} />
    </div>
  );
}
