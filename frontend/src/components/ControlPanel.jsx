import React from 'react';

export default function ControlPanel({ isProcessing, setIsProcessing, forceFinalize }) {
  return (
    <div className="p-6 rounded-2xl bg-slate-800/60 backdrop-blur-md border border-slate-700/50 shadow-lg">
      <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4">Controls</h3>
      
      <div className="flex flex-col sm:flex-row gap-4">
        <button
          onClick={() => setIsProcessing(!isProcessing)}
          className={`flex-1 py-3 px-6 rounded-xl font-bold tracking-wide transition-all shadow-lg active:scale-95 ${
            isProcessing 
              ? 'bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20 hover:shadow-red-500/20' 
              : 'bg-green-500/10 text-green-400 border border-green-500/30 hover:bg-green-500/20 hover:shadow-green-500/20'
          }`}
        >
          {isProcessing ? 'Pause Inference' : 'Start Inference'}
        </button>
        
        <button
          onClick={forceFinalize}
          className="flex-1 py-3 px-6 rounded-xl font-bold tracking-wide text-brand-primary bg-brand-primary/10 border border-brand-primary/30 hover:bg-brand-primary/20 transition-all shadow-lg hover:shadow-brand-primary/20 active:scale-95"
        >
          Finalize Sentence Now
        </button>
      </div>
    </div>
  );
}
