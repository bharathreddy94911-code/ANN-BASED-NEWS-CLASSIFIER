import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Database, 
  Sparkles, 
  PlayCircle, 
  FlaskConical, 
  BarChart3, 
  Info 
} from 'lucide-react';

const Sidebar = () => {
  const links = [
    { to: '/', label: 'Dashboard', icon: LayoutDashboard },
    { to: '/dataset', label: 'Dataset Explorer', icon: Database },
    { to: '/classifier', label: 'News Classifier', icon: Sparkles },
    { to: '/training', label: 'Model Training', icon: PlayCircle },
    { to: '/experiments', label: 'Experiments', icon: FlaskConical },
    { to: '/evaluation', label: 'Evaluation', icon: BarChart3 },
    { to: '/about', label: 'About Project', icon: Info },
  ];

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col justify-between shrink-0">
      <div>
        <div className="p-6 border-b border-slate-800 flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 font-bold">
            AG
          </div>
          <div>
            <h1 className="text-sm font-semibold text-white tracking-wide">AG News Classifier</h1>
            <p className="text-xs text-slate-400">CBOW + MLP System</p>
          </div>
        </div>

        <nav className="p-4 space-y-1">
          {links.map((link) => {
            const Icon = link.icon;
            return (
              <NavLink
                key={link.to}
                to={link.to}
                className={({ isActive }) =>
                  "flex items-center space-x-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors " +
                  (isActive
                    ? "bg-cyan-500/15 text-cyan-400 border border-cyan-500/20"
                    : "text-slate-300 hover:bg-slate-800/60 hover:text-white")
                }
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span>{link.label}</span>
              </NavLink>
            );
          })}
        </nav>
      </div>

      <div className="p-4 m-4 rounded-xl bg-slate-950/60 border border-slate-800/80 text-xs">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-slate-400 font-medium">Dataset Status</span>
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            Mikami063
          </span>
        </div>
        <p className="text-slate-400 text-[11px] leading-relaxed">
          4 Classes: World, Sports, Business, Sci/Tech
        </p>
      </div>
    </aside>
  );
};

export default Sidebar;
