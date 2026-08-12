import { Routes, Route } from "react-router-dom";
import { Sidebar } from "./components/Sidebar";
import { ChatPanel } from "./components/ChatPanel";
import { Dashboard } from "./pages/Dashboard";
import { Models } from "./pages/Models";
import { NewRun } from "./pages/NewRun";
import { Runs } from "./pages/Runs";
import { RunDetail } from "./pages/RunDetail";
import { Compare } from "./pages/Compare";
import { Workers } from "./pages/Workers";

export default function App() {
  return (
    <div className="flex min-h-screen bg-bg">
      <Sidebar />
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
          </Routes>
        </div>
      </main>
      <ChatPanel />
    </div>
  );
}
