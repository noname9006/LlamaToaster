/** Progress of the job a machine is running: a fit map reports "N of ~M
 * calls" in its detail, everything else counts items. */
export function jobPercent(detail: string | undefined, itemIdx: number | undefined, itemsTotal: number | undefined): number {
  const m = detail ? /\((\d+)\/(\d+)\)|(\d+) of ~(\d+) calls/.exec(detail) : null;
  if (m) {
    const done = Number(m[1] ?? m[3]);
    const total = Number(m[2] ?? m[4]);
    if (total > 0) return Math.round((done / total) * 100);
  }
  return itemsTotal ? Math.round(((itemIdx ?? 0) / itemsTotal) * 100) : 10;
}
