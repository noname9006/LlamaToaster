import { useEffect, useMemo, useState } from "react";

// Log-scale slider: an internal linear 0-1000 position maps onto 1-200 tok/s
// so the low end (where the difference between e.g. 2 and 8 tok/s is huge)
// gets as much of the slider's travel as the high end (where 120 vs 180
// barely matters).
const SLIDER_MAX = 1000;
const TPS_MIN = 1;
const TPS_MAX = 200;
const LOG_MIN = Math.log(TPS_MIN);
const LOG_MAX = Math.log(TPS_MAX);

function sliderToTps(t: number): number {
  const frac = t / SLIDER_MAX;
  return Math.min(TPS_MAX, Math.max(TPS_MIN, Math.round(Math.exp(LOG_MIN + frac * (LOG_MAX - LOG_MIN)))));
}

function tpsToSlider(tps: number): number {
  const frac = (Math.log(tps) - LOG_MIN) / (LOG_MAX - LOG_MIN);
  return Math.round(frac * SLIDER_MAX);
}

const DEFAULT_TPS = 15;

// Each band's reply is long enough (~350-500 chars) to still visibly animate
// even at 200 tok/s, and actually answers the card's title instead of
// standing in as lorem-ipsum filler.
const DEMO_BANDS: { max: number; reply: string }[] = [
  {
    max: 3,
    reply:
      "At this speed you're watching the model think. Each word lands roughly a second or more apart — slower than a person typing on a phone. It's the pace you'd expect from a large model squeezed onto hardware that's mostly out of headroom, running on CPU with no room to spare. Fine for a batch job you can walk away from; painful for a live conversation.",
  },
  {
    max: 10,
    reply:
      "This is close to how fast a person reads out loud — comfortable, but you'll still notice each word arriving rather than the whole sentence appearing at once. It's typical for a large model doing real CPU work, or a mid-size model split between a modest GPU and system RAM. Usable for chat, though you'll find yourself waiting a beat before replying.",
  },
  {
    max: 30,
    reply:
      "This is roughly average silent-reading speed — text keeps pace with your eyes, so it feels like a natural conversation instead of a countdown. Most well-tuned mid-size models on a decent consumer GPU land somewhere around here. This is the speed most people would call 'good enough' for everyday chat, coding help, or drafting.",
  },
  {
    max: 80,
    reply:
      "This is faster than almost anyone reads — the text is keeping up with you, not the other way around. You'd typically see this from a smaller model fully offloaded to a strong GPU, or a larger one running efficiently with good batching. The bottleneck stops being the model and starts being how fast you can actually process what it's saying.",
  },
  {
    max: Infinity,
    reply:
      "This is well beyond reading speed — words appear in a blur, and a long reply feels instantaneous. You'd see numbers like this from a small, fast model on a high-end GPU, or from heavily optimized inference on short generations. It's less about following along live and more about throughput: bulk generation, summarizing many documents, or powering an app rather than a person watching the screen.",
  },
];

// One fake "token" per word -- a rough but readable stand-in for real
// sub-word tokenization, good enough for a speed-feel demo that is
// explicitly not a real generation.
export function TokSpeedDemo() {
  const [sliderT, setSliderT] = useState(() => tpsToSlider(DEFAULT_TPS));
  const [revealed, setRevealed] = useState(0);

  const tps = sliderToTps(sliderT);
  const band = DEMO_BANDS.find((b) => tps <= b.max)!;
  const words = useMemo(() => band.reply.split(" "), [band.reply]);

  useEffect(() => {
    let i = 0;
    setRevealed(0);
    // Hold the finished reply on screen for ~1.5s (in ticks, since the tick
    // rate itself scales with tps) before looping back to the start.
    const pauseTicks = Math.max(3, Math.round(tps * 1.5));
    const id = window.setInterval(() => {
      i++;
      if (i <= words.length) {
        setRevealed(i);
      } else if (i >= words.length + pauseTicks) {
        i = 0;
        setRevealed(0);
      }
    }, 1000 / tps);
    return () => window.clearInterval(id);
  }, [tps, words]);

  const done = revealed >= words.length;

  return (
    <div className="rounded-xl border border-border bg-surface p-5">
      <h3 className="text-lg font-semibold text-fg">What does {tps} tok/s mean?</h3>
      <p className="mt-0.5 text-xs text-muted">Simulated for illustration — not a real generation.</p>

      <div className="mt-4">
        <div className="flex items-center justify-between text-xs text-muted">
          <span>{TPS_MIN} tok/s</span>
          <span className="text-sm font-semibold text-fg">{tps} tok/s</span>
          <span>{TPS_MAX} tok/s</span>
        </div>
        <input
          type="range"
          min={0}
          max={SLIDER_MAX}
          value={sliderT}
          onChange={(e) => setSliderT(Number(e.target.value))}
          className="mt-1.5 w-full"
          aria-label="Generation speed in tokens per second"
        />
      </div>

      <div className="mt-4 flex flex-col gap-2 rounded-lg border border-border bg-surface-raised p-3">
        <div className="flex justify-end">
          <div className="max-w-[85%] rounded-lg rounded-br-sm bg-accent/15 px-3 py-2 text-sm text-accent">
            What does {tps} tok/s mean?
          </div>
        </div>
        <div className="flex justify-start">
          <div className="max-w-[85%] rounded-lg rounded-bl-sm bg-surface px-3 py-2 text-sm text-fg">
            {words.slice(0, revealed).join(" ")}
            {!done && <span className="animate-pulse">▋</span>}
          </div>
        </div>
      </div>
    </div>
  );
}
