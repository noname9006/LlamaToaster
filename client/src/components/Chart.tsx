import { useEffect, useRef } from "react";
import { Chart as ChartJS, type ChartConfiguration } from "chart.js/auto";

// Thin canvas wrapper around Chart.js (same library the old server-rendered
// pages used via CDN script tag) -- recreates the chart whenever the config
// changes, mirroring the previous destroy-then-recreate-on-reload behavior.
export function Chart({ config }: { config: ChartConfiguration }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<ChartJS | null>(null);
  const configKey = JSON.stringify(config);

  useEffect(() => {
    if (!canvasRef.current) return;
    chartRef.current?.destroy();
    chartRef.current = new ChartJS(canvasRef.current, config);
    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
    // Re-run only when the serialized config actually changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [configKey]);

  return <canvas ref={canvasRef} />;
}
