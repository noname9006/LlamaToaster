import { Navigate, Route, Routes } from "react-router-dom";
import { Overview } from "./pages/Overview";
import { TestPage } from "./pages/TestPage";
import { BreadMarks } from "./pages/BreadMarks";

// Screens: the overview (platform stats, settings, every user's tests), the
// bread fit mark style, and a per-test page that renders any one of them the way its owner sees it. The
// server falls back to this bundle's index.html for any non-/api path on the
// admin hostname (server/src/index.ts), so /tests/:id works as a deep link.
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Overview />} />
      <Route path="/tests/:id" element={<TestPage />} />
      <Route path="/bread-marks" element={<BreadMarks />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
