import { useEffect, useState } from "react";
import { api } from "./client";
import type { RocmSupportList } from "../types";

// The server's daily-refreshed list of AMD GPUs ROCm supports (see
// server/src/rocm-support.ts) -- the input detectBackend needs to decide
// "rocm or vulkan" for a specific GPU. Read once per page load and shared by
// every caller; undefined until it arrives, or if the fetch fails, in which
// case detectBackend just uses its built-in snapshot -- a missing list never
// blocks the page.
let inflight: Promise<RocmSupportList | undefined> | null = null;

function loadRocmSupport(): Promise<RocmSupportList | undefined> {
  if (!inflight) {
    inflight = api
      .getRocmSupport()
      .then((snap) => snap.list)
      .catch(() => {
        inflight = null; // let a later mount retry
        return undefined;
      });
  }
  return inflight;
}

export function useRocmSupport(): RocmSupportList | undefined {
  const [list, setList] = useState<RocmSupportList | undefined>(undefined);
  useEffect(() => {
    let cancelled = false;
    void loadRocmSupport().then((l) => {
      if (!cancelled) setList(l);
    });
    return () => {
      cancelled = true;
    };
  }, []);
  return list;
}
