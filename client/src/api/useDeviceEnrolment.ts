import { useEffect, useState } from "react";
import { api } from "./client";
import type { DeviceStatusResponse } from "../types";

// Matches server/src/session.ts's generateUserCode (4 chars, a dash, 4 more,
// drawn from Crockford base32 minus ambiguous characters) -- used both to
// know when the field is "full enough to poll" and to reject obviously
// unfinished input before it ever reaches the server.
const CODE_RE = /^[A-Z0-9]{4}-[A-Z0-9]{4}$/;

function normalizeCodeInput(raw: string): string {
  const upper = raw
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, "")
    .slice(0, 8);
  return upper.length > 4 ? `${upper.slice(0, 4)}-${upper.slice(4)}` : upper;
}

const POLL_INTERVAL_MS = 2000;

// Enrolment code entry + polling + approve/deny, extracted out of
// AddMachinePanel so a caller (Workers.tsx) can render a compact code box
// elsewhere on the page -- e.g. next to the collapsed "Add a machine"
// summary -- while it stays wired to the very same state as the full panel
// underneath it.
export function useDeviceEnrolment() {
  const [codeInput, setCodeInput] = useState("");
  const [status, setStatus] = useState<DeviceStatusResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [approved, setApproved] = useState<{ hostname: string; merged: boolean } | null>(null);

  const validCode = CODE_RE.test(codeInput);

  // Polls GET /api/device/status while a full-length code is present (§3.1
  // step 4) -- stops as soon as it resolves to "approved" (whether from this
  // tab's own Approve click below, or a second browser tab/device that
  // approved it first) or the input changes to something no longer a full
  // code.
  useEffect(() => {
    if (!validCode) return;
    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | null = null;
    async function poll() {
      try {
        const s = await api.getDeviceStatus(codeInput);
        if (cancelled) return;
        setStatus(s);
        setError(null);
        if (s.state === "approved" && timer) {
          clearInterval(timer);
          timer = null;
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      }
    }
    void poll();
    timer = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [validCode, codeInput]);

  function handleCodeChange(raw: string) {
    setCodeInput(normalizeCodeInput(raw));
    setStatus(null);
    setError(null);
  }

  async function handleApprove(mergeInto?: string) {
    setBusy(true);
    setError(null);
    try {
      // confirm_duplicate reflects what the last status poll already showed
      // -- see server/src/routes/device.ts's POST /api/device/approve and
      // workerRepo.findPossibleDuplicate's doc comment for why this exists:
      // deleting a worker's install folder wipes its persisted machine_id,
      // so re-running setup looks like a brand-new machine to the server,
      // and without this check would silently create an indistinguishable
      // duplicate of a machine the user already has. mergeInto instead asks
      // the server to re-attach this connection to that existing machine.
      const confirmDuplicate = !mergeInto && status?.state === "pending" && status.possibleDuplicate != null;
      const res = await api.approveDevice(codeInput, confirmDuplicate, mergeInto);
      if (!res.ok) {
        // Race: the duplicate was only detected server-side just now (this
        // poll hadn't caught up yet) -- the next poll tick will pick up
        // possibleDuplicate and relabel the buttons; nothing was approved.
        setError('This machine looks like one you already have -- pick "Merge" or "Add as new machine" once it appears below.');
        return;
      }
      setApproved({ hostname: res.machine.hostname ?? "This machine", merged: res.merged === true });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  // No dedicated deny endpoint exists (or needs to) -- a code nobody approves
  // just expires on its own in 15 minutes (server/src/routes/device.ts's
  // ENROLMENT_TTL_MS). This just walks the user away from the confirm card.
  function handleDeny() {
    setCodeInput("");
    setStatus(null);
    setError(null);
  }

  return {
    codeInput,
    handleCodeChange,
    validCode,
    status,
    busy,
    error,
    approved,
    handleApprove,
    handleDeny,
  };
}
