import React from 'react';

export default function ControlPanel({ isProcessing, setIsProcessing, forceFinalize }) {
  return (
    <div className="bg-[#09090b] p-4 rounded-xl border border-zinc-800 shadow-sm flex gap-4 items-center">
      <button
        onClick={() => setIsProcessing(!isProcessing)}
        className={`flex-1 py-2.5 px-4 rounded text-sm font-semibold transition-colors flex items-center justify-center gap-2 border ${
          isProcessing 
            ? 'bg-rose-500/10 text-rose-400 border-rose-500/30 hover:bg-rose-500/20' 
            : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/20'
        }`}
      >
        {isProcessing ? 'Pause Inference' : 'Start Inference'}
      </button>
      
      <button
        onClick={forceFinalize}
        className="flex-1 py-2.5 px-4 rounded text-sm font-semibold transition-colors border bg-zinc-900 border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:text-white flex items-center justify-center gap-2"
      >
        Finalize Sentence
      </button>
    </div>
  );
}
