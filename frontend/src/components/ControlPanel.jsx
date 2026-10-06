import React from 'react';

export default function ControlPanel({ isProcessing, setIsProcessing, forceFinalize }) {
  return (
    <div className="glass-card p-6 rounded-3xl flex flex-col sm:flex-row gap-4 items-center justify-between">
      <div className="flex-1 w-full">
        <button
          onClick={() => setIsProcessing(!isProcessing)}
          className={`w-full py-4 px-8 rounded-2xl font-bold tracking-wide transition-all duration-300 shadow-xl flex items-center justify-center gap-3 ${
            isProcessing 
              ? 'bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20 hover:shadow-red-500/20' 
              : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/20 hover:shadow-emerald-500/20'
          }`}
        >
          {isProcessing ? (
            <>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Pause Inference
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Start Inference
            </>
          )}
        </button>
      </div>
      
      <div className="flex-1 w-full">
        <button
          onClick={forceFinalize}
          className="w-full py-4 px-8 rounded-2xl font-bold tracking-wide text-brand-secondary bg-brand-secondary/10 border border-brand-secondary/30 hover:bg-brand-secondary/20 transition-all duration-300 shadow-xl hover:shadow-brand-secondary/20 flex items-center justify-center gap-3"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
          </svg>
          Finalize Sentence
        </button>
      </div>
    </div>
  );
}
