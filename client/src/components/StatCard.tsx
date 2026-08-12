export function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex-1 rounded-xl border border-border bg-surface px-6 py-4 text-center">
      <div className="text-3xl font-bold text-accent">{value}</div>
      <div className="mt-1 text-sm text-muted">{label}</div>
    </div>
  );
}
