import React from 'react';

export default function SignDisplay({ currentSign, confidence }) {
  const isRecognized = currentSign && currentSign !== "None";
  
  return (
    <div className="bg-[#09090b] p-6 rounded-xl border border-zinc-800 shadow-sm flex flex-col">
      <h2 className="text-sm font-semibold text-zinc-400 mb-6">Current Prediction</h2>
      
      <div className="flex flex-col items-center justify-center flex-1 min-h-[140px] bg-zinc-900/50 rounded-lg border border-zinc-800 p-6">
        {isRecognized ? (
          <div className="flex flex-col items-center w-full">
            <div className="text-5xl font-bold text-white tracking-tight mb-6 drop-shadow-sm">
              {currentSign}
            </div>
            
            <div className="w-full max-w-[200px] flex items-center gap-3">
              <span className="text-xs font-medium text-zinc-500">Conf.</span>
              <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-indigo-500 transition-all duration-300 ease-out"
                  style={{ width: `${Math.min(100, Math.max(0, confidence * 100))}%` }}
                ></div>
              </div>
              <span className="text-xs font-mono font-medium text-zinc-400">
                {(confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        ) : (
          <div className="text-zinc-600 text-sm font-medium flex flex-col items-center gap-2">
            <svg className="w-6 h-6 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            Awaiting Sign...
          </div>
        )}
      </div>
    </div>
  );
}
