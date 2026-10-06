import React from 'react';

export default function SignDisplay({ currentSign, confidence }) {
  const isRecognized = currentSign && currentSign !== "None";
  
  return (
    <div className="glass-panel p-8 rounded-3xl relative overflow-hidden group">
      {/* Dynamic Background Glow based on recognition */}
      <div className={`absolute -top-24 -right-24 w-64 h-64 rounded-full blur-[80px] transition-all duration-1000 ${isRecognized ? 'bg-brand-primary/40' : 'bg-slate-700/20'}`}></div>
      
      <div className="flex items-center justify-between mb-6 relative z-10">
        <h2 className="text-xs font-bold text-slate-400 uppercase tracking-[0.2em]">Active Sign</h2>
        <svg className="w-5 h-5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
        </svg>
      </div>
      
      <div className="flex flex-col items-center justify-center min-h-[160px] relative z-10">
        {isRecognized ? (
          <div className="flex flex-col items-center w-full">
            <div className="text-6xl md:text-7xl font-black text-white tracking-tight mb-8 drop-shadow-[0_0_15px_rgba(255,255,255,0.3)] transform transition-transform group-hover:scale-105">
              {currentSign}
            </div>
            
            <div className="w-full max-w-[240px] space-y-2">
              <div className="flex justify-between text-xs font-medium text-slate-400">
                <span>Confidence</span>
                <span className="text-brand-secondary">{(confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden shadow-inner">
                <div 
                  className="h-full bg-gradient-to-r from-brand-secondary to-brand-primary relative"
                  style={{ width: `${Math.min(100, Math.max(0, confidence * 100))}%` }}
                >
                  <div className="absolute inset-0 bg-white/20 animate-[slide-right_2s_ease-in-out_infinite]"></div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4 text-slate-500">
            <div className="w-16 h-16 rounded-full border-2 border-slate-700/50 border-t-slate-500 animate-spin"></div>
            <p className="text-sm font-medium tracking-wide">Awaiting input...</p>
          </div>
        )}
      </div>
    </div>
  );
}
