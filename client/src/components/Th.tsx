import type { ReactNode } from "react";
import { Tooltip } from "./Tooltip";

export interface SortState {
  key: string;
  dir: "asc" | "desc";
}

interface ThProps {
  label: ReactNode;
  description?: string;
  sortKey?: string;
  sort?: SortState | null;
  onSort?: (key: string) => void;
  className?: string;
}

// One header cell for every sortable/described table in the app (Runs.tsx,
// RunDetail.tsx's two tables) -- hover shows `description` via Tooltip,
// clicking toggles sort via `onSort` when `sortKey` is given. `sortKey`/
// `onSort` are plain strings rather than a generic <K> because TSX doesn't
// support explicit type arguments on JSX tags -- each caller just narrows in
// its own onSort callback instead.
export function Th({ label, description, sortKey, sort, onSort, className }: ThProps) {
  const sortable = sortKey !== undefined && !!onSort;
  const active = sortable && sort?.key === sortKey;
  const glyph = active ? (sort!.dir === "asc" ? "▲" : "▼") : "⇅";

  const content = (
    <span
      className={`inline-flex items-center gap-0.5 ${
        sortable ? "cursor-pointer select-none hover:text-fg" : ""
      } ${active ? "text-fg" : ""}`}
      onClick={sortable ? () => onSort!(sortKey!) : undefined}
    >
      {label}
      {sortable && <span className={active ? "text-accent" : "text-muted/60"}>{glyph}</span>}
    </span>
  );

  return (
    <th className={`px-3 py-2 font-medium whitespace-nowrap ${className ?? ""}`}>
      {description ? <Tooltip text={description}>{content}</Tooltip> : content}
    </th>
  );
}

export function toggleSort(current: SortState | null, key: string, defaultDir: "asc" | "desc" = "asc"): SortState {
  if (current?.key === key) return { key, dir: current.dir === "asc" ? "desc" : "asc" };
  return { key, dir: defaultDir };
}
