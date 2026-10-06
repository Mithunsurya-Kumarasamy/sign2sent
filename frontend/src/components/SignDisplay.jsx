import React from 'react';

export default function SignDisplay({ currentSign, confidence }) {
  const isRecognized = currentSign && currentSign !== "None";
  
  return (
    <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col">
      <h2 className="text-sm font-semibold text-slate-700 mb-6">Current Prediction</h2>
      
      <div className="flex flex-col items-center justify-center flex-1 min-h-[140px] bg-slate-50 rounded-lg border border-slate-100 p-6">
        {isRecognized ? (
          <div className="flex flex-col items-center w-full">
            <div className="text-5xl font-extrabold text-indigo-600 tracking-tight mb-6">
              {currentSign}
            </div>
            
            <div className="w-full max-w-[200px] flex items-center gap-3">
              <span className="text-xs font-medium text-slate-500">Conf.</span>
              <div className="flex-1 h-2 bg-slate-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-indigo-500 transition-all duration-300 ease-out"
                  style={{ width: `${Math.min(100, Math.max(0, confidence * 100))}%` }}
                ></div>
              </div>
              <span className="text-xs font-mono font-medium text-slate-700">
                {(confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        ) : (
          <div className="text-slate-400 text-sm font-medium flex flex-col items-center gap-2">
            <svg className="w-6 h-6 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            Awaiting Sign...
          </div>
        )}
      </div>
    </div>
  );
}
