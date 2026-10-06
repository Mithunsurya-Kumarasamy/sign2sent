import React from 'react';

export default function SignDisplay({ currentSign, confidence }) {
  const isRecognized = currentSign && currentSign !== "None";
  
  return (
    <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-800/80 to-slate-900/80 backdrop-blur-xl border border-slate-700/50 shadow-2xl relative overflow-hidden group">
      <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 group-hover:bg-blue-400/20 transition-all duration-700"></div>
      
      <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4">Current Prediction</h2>
      
      <div className="flex flex-col items-center justify-center py-6 min-h-[160px]">
        {isRecognized ? (
          <>
            <div className="text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-400 animate-[pulse_2s_ease-in-out_infinite] mb-4">
              {currentSign}
            </div>
            
            <div className="flex items-center gap-3 w-full max-w-[200px]">
              <div className="h-1.5 w-full bg-slate-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 transition-all duration-300"
                  style={{ width: `${Math.min(100, Math.max(0, confidence * 100))}%` }}
                ></div>
              </div>
              <span className="text-xs font-mono text-slate-400">
                {(confidence * 100).toFixed(0)}%
              </span>
            </div>
          </>
        ) : (
          <div className="text-2xl font-light text-slate-500 italic">
            Waiting for sign...
          </div>
        )}
      </div>
    </div>
  );
}
