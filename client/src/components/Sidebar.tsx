import { NavLink } from "react-router-dom";
import type { ComponentType, SVGProps } from "react";
import { IconGrid, IconBox, IconPlusCircle, IconList, IconBarChart, IconServer } from "./icons";

interface NavItem {
  to: string;
  label: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  end?: boolean;
}

const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Dashboard", icon: IconGrid, end: true },
  { to: "/models", label: "Models", icon: IconBox },
  { to: "/new-run", label: "New Run", icon: IconPlusCircle },
  { to: "/runs", label: "Runs", icon: IconList },
  { to: "/compare", label: "Compare", icon: IconBarChart },
  { to: "/workers", label: "Workers", icon: IconServer },
];

export function Sidebar() {
  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-border bg-surface">
      <div className="flex items-center gap-2 px-5 py-5">
        <img src="/logo.png" alt="LlamaToaster" className="w-full" />
      </div>
      <nav className="flex flex-1 flex-col gap-0.5 px-3">
        {NAV_ITEMS.map(({ to, label, icon: ItemIcon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                isActive ? "bg-accent/15 text-accent" : "text-muted hover:bg-white/5 hover:text-fg"
              }`
            }
          >
            <ItemIcon className="h-4.5 w-4.5" width={18} height={18} />
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="px-5 py-4 text-xs text-muted">Tailscale-only orchestrator</div>
    </aside>
  );
}
