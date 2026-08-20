import { NavLink } from "react-router-dom";
import type { ComponentType, SVGProps } from "react";
import { IconGrid, IconBox, IconPlusCircle, IconList, IconBarChart, IconServer, IconSettings } from "./icons";

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

// Kept separate from NAV_ITEMS -- Settings is account-scoped (session
// management, connected accounts), not part of the benchmarking workflow
// the rest of the nav covers, so it gets its own spot below the divider
// rather than blending into that list.
const SETTINGS_ITEM: NavItem = { to: "/settings", label: "Settings", icon: IconSettings };

interface SidebarProps {
  // Settings' own routes (GET/DELETE /api/sessions, GET /api/auth/identities)
  // are only ever REGISTERED server-side when AUTH_ENABLED (server/src/index.ts)
  // -- showing this link the rest of the time would point at routes that
  // 404, not a page that gracefully explains itself. Hidden rather than
  // shown-disabled since there's nothing here for a Stage-1-only deployment
  // to grow into yet.
  authEnabled: boolean;
}

export function Sidebar({ authEnabled }: SidebarProps) {
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
      {authEnabled && (
        <div className="border-t border-border px-3 py-2">
          <NavLink
            to={SETTINGS_ITEM.to}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                isActive ? "bg-accent/15 text-accent" : "text-muted hover:bg-white/5 hover:text-fg"
              }`
            }
          >
            <IconSettings className="h-4.5 w-4.5" width={18} height={18} />
            {SETTINGS_ITEM.label}
          </NavLink>
        </div>
      )}
    </aside>
  );
}
