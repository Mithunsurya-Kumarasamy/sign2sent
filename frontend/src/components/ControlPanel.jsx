import React from 'react';

export default function ControlPanel({ isProcessing, setIsProcessing, forceFinalize }) {
  return (
    <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex gap-4 items-center">
      <button
        onClick={() => setIsProcessing(!isProcessing)}
        className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-semibold transition-colors flex items-center justify-center gap-2 border ${
          isProcessing 
            ? 'bg-rose-50 text-rose-700 border-rose-200 hover:bg-rose-100' 
            : 'bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100'
        }`}
      >
        {isProcessing ? 'Pause Inference' : 'Start Inference'}
      </button>
      
      <button
        onClick={forceFinalize}
        className="flex-1 py-2.5 px-4 rounded-lg text-sm font-semibold transition-colors border bg-white border-indigo-200 text-indigo-700 hover:bg-indigo-50 flex items-center justify-center gap-2"
      >
        Finalize Sentence
      </button>
    </div>
  );
}
