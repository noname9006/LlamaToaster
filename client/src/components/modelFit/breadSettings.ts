// Whether the Models page draws bread marks, and in what style: the
// operator's AppSettings (admin console), read once per page load from
// /api/auth/status. Until it arrives, and if it fails, the shipped default
// (bread on, DEFAULT_BREAD_MARK_STYLE) applies.

import { useEffect, useState } from "react";
import { api } from "../../api/client";
import { DEFAULT_BREAD_MARK_STYLE, type BreadMarkStyle } from "../../../../shared/breadMarks.js";

export interface BreadSettings {
  enabled: boolean;
  style: BreadMarkStyle;
}

const DEFAULT: BreadSettings = { enabled: true, style: DEFAULT_BREAD_MARK_STYLE };

let current: BreadSettings = DEFAULT;
let loading: Promise<void> | null = null;
const listeners = new Set<(s: BreadSettings) => void>();

function load(): Promise<void> {
  loading ??= api
    .getAuthStatus()
    .then((status) => {
      const a = status.appSettings;
      current = { enabled: a?.breadMarksEnabled ?? true, style: a?.breadMarkStyle ?? DEFAULT_BREAD_MARK_STYLE };
      for (const l of listeners) l(current);
    })
    .catch(() => {});
  return loading;
}

export function useBreadSettings(): BreadSettings {
  const [s, setS] = useState(current);
  useEffect(() => {
    listeners.add(setS);
    void load();
    setS(current);
    return () => {
      listeners.delete(setS);
    };
  }, []);
  return s;
}
