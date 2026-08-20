import { useEffect, useState } from "react";
import { Routes, Route, useLocation, useNavigate } from "react-router-dom";
import { Sidebar } from "./components/Sidebar";
import { ChatPanel } from "./components/ChatPanel";
import { Dashboard } from "./pages/Dashboard";
import { Models } from "./pages/Models";
import { NewRun } from "./pages/NewRun";
import { Runs } from "./pages/Runs";
import { RunDetail } from "./pages/RunDetail";
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
          appSettings: { communitySharingAllowed: false, accountDeletionAllowed: true },
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

  return (
    <div className="flex min-h-screen bg-bg">
      <Sidebar authEnabled={authStatus.authEnabled} />
      <main className="h-screen flex-1 overflow-y-auto">
        <div className="w-full px-8 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/models" element={<Models />} />
            <Route path="/new-run" element={<NewRun />} />
            <Route path="/runs" element={<Runs />} />
            <Route path="/runs/:id" element={<RunDetail />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/workers" element={<Workers />} />
            <Route path="/device" element={<Device />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </div>
      </main>
      <ChatPanel />
    </div>
  );
}
