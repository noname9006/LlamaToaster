import { NavLink, useLocation } from "react-router-dom";
import { LtIcon, type LtIconName } from "./ltIcons";

interface NavItem {
  to: string;
  label: string;
  icon: LtIconName;
  end?: boolean;
  /** Other routes that light this item up (a detail page under its list). */
  also?: string[];
}

// The v2 navigation (docs/plans/app-v2.dc.html). "New test" is the
// optimization flow; Custom Test hangs under it, a test's detail page under
// Tests. Settings is account-scoped and only exists when auth is on.
export const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Dashboard", icon: "grid", end: true },
  { to: "/benchmark", label: "New test", icon: "plus", also: ["/custom-test", "/new"] },
  { to: "/tests", label: "Tests", icon: "list" },
  { to: "/compare", label: "Compare", icon: "bars" },
  { to: "/models", label: "Models", icon: "box" },
  { to: "/workers", label: "Machines", icon: "server", also: ["/device"] },
];

const SETTINGS_ITEM: NavItem = { to: "/settings", label: "Settings", icon: "sliders" };

export function navItemsFor(authEnabled: boolean): NavItem[] {
  return authEnabled ? [...NAV_ITEMS, SETTINGS_ITEM] : NAV_ITEMS;
}

function isActive(item: NavItem, pathname: string): boolean {
  if (item.end) return pathname === item.to;
  return [item.to, ...(item.also ?? [])].some((p) => pathname === p || pathname.startsWith(p + "/"));
}

export function NavList({ authEnabled, onNavigate, large }: { authEnabled: boolean; onNavigate?: () => void; large?: boolean }) {
  const { pathname } = useLocation();
  return (
    <>
      {navItemsFor(authEnabled).map((item) => {
        const active = isActive(item, pathname);
        return (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            onClick={onNavigate}
            aria-current={active ? "page" : undefined}
            className={`flex w-full items-center gap-3 border-l-2 px-3 text-sm font-medium transition-colors ${
              large ? "min-h-11" : "min-h-10"
            } ${active ? "border-accent bg-accent-tint text-accent" : "border-transparent text-fg-2 hover:bg-surface-raised hover:text-fg"}`}
          >
            <LtIcon name={item.icon} className="flex-none" />
            {item.label}
          </NavLink>
        );
      })}
    </>
  );
}

interface SidebarProps {
  // Settings' own routes are only registered server-side when AUTH_ENABLED
  // -- see server/src/index.ts -- so the link is hidden the rest of the time.
  authEnabled: boolean;
  footer?: string | null;
}

export function Sidebar({ authEnabled, footer }: SidebarProps) {
  return (
    <aside className="sticky top-0 flex h-screen w-56 flex-none flex-col border-r border-border bg-surface">
      <div className="flex items-center gap-2.5 px-[18px] pb-5 pt-[18px]">
        <img src="/toaster_favicon.png" alt="" className="block h-8 w-8 object-contain" />
        <span className="font-display text-lg font-semibold">LlamaToaster</span>
      </div>
      <nav aria-label="Main" className="flex flex-col gap-0.5 px-2.5">
        <NavList authEnabled={authEnabled} />
      </nav>
      {footer && (
        <p className="mt-auto truncate border-t border-border px-[18px] py-4 font-mono text-xs text-muted">{footer}</p>
      )}
    </aside>
  );
}
