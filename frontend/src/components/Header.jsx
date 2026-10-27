import React from 'react';
import { RefreshCw, Activity } from 'lucide-react';

const Header = ({ title, description, isBusy, onRefresh }) => {
  return (
    <header className="h-16 px-8 border-b border-slate-800 bg-slate-900/60 backdrop-blur flex items-center justify-between shrink-0">
      <div>
        <h2 className="text-base font-semibold text-white tracking-tight">{title}</h2>
        {description && <p className="text-xs text-slate-400 mt-0.5">{description}</p>}
      </div>
      <div className="flex items-center space-x-3">
        {isBusy && (
          <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs font-medium animate-pulse">
            <Activity className="w-3.5 h-3.5" />
            <span>Training Active</span>
          </div>
        )}
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="p-2 rounded-lg bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 transition-colors"
            title="Refresh page data"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        )}
      </div>
    </header>
  );
};

export default Header;
