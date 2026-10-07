import React, { useState } from 'react';

export default function ControlPanel({
  isProcessing,
  setIsProcessing,
  forceFinalize,
  onClear,
  confidenceThreshold,
  setConfidenceThreshold,
  stableFramesRequired,
  setStableFramesRequired,
  pauseFramesRequired,
  setPauseFramesRequired
}) {
  const [showSettings, setShowSettings] = useState(false);

  return (
    <div className="bg-[#09090b] p-4 rounded-xl border border-zinc-800 shadow-sm flex flex-col gap-3">
      {/* Primary Action Buttons */}
      <div className="flex flex-wrap gap-3 items-center">
        <button
          onClick={() => setIsProcessing(!isProcessing)}
          className={`flex-1 min-w-[140px] py-2.5 px-4 rounded text-sm font-semibold transition-colors flex items-center justify-center gap-2 border cursor-pointer ${isProcessing
              ? 'bg-rose-500/10 text-rose-400 border-rose-500/30 hover:bg-rose-500/20'
              : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/20'
            }`}
        >
          {isProcessing ? (
            <>
              <span className="w-2 h-2 rounded-full bg-rose-400 animate-pulse"></span>
              Pause Inference
            </>
          ) : (
            <>
              <svg className="w-4 h-4 text-emerald-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
              </svg>
              Start Inference
            </>
          )}
        </button>

        <button
          onClick={forceFinalize}
          className="flex-1 min-w-[140px] py-2.5 px-4 rounded text-sm font-semibold transition-colors border bg-zinc-900 border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:text-white flex items-center justify-center gap-2 cursor-pointer"
        >
          <svg className="w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Finalize Sentence
        </button>

        <button
          onClick={onClear}
          className="py-2.5 px-3 rounded text-sm font-medium transition-colors border bg-zinc-900 border-zinc-800 text-zinc-400 hover:text-rose-400 hover:border-rose-900/50 flex items-center justify-center gap-1.5 cursor-pointer"
          title="Reset sequence and predictions"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Reset All
        </button>

        <button
          onClick={() => setShowSettings(!showSettings)}
          className={`p-2.5 rounded text-sm transition-colors border cursor-pointer ${showSettings
              ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300'
              : 'bg-zinc-900 border-zinc-800 text-zinc-400 hover:text-zinc-200'
            }`}
          title="Toggle Debounce & Stability Settings"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </button>
      </div>

      {/* Configurable Stability & Word-Boundary Settings Panel */}
      {showSettings && (
        <div className="pt-3 border-t border-zinc-800 text-xs flex flex-col gap-3 bg-zinc-950/40 p-3 rounded-lg">
          <div className="flex justify-between items-center text-zinc-300 font-semibold">
            <span>Debouncing & Stability Configuration</span>
            <span className="text-[11px] text-zinc-500 font-normal">Active frame interval: 250ms</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {/* Confidence Threshold */}
            <div className="flex flex-col gap-1">
              <div className="flex justify-between text-zinc-400">
                <span>Min Confidence:</span>
                <span className="font-mono text-indigo-400 font-medium">{(confidenceThreshold * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.50"
                max="0.95"
                step="0.05"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                className="accent-indigo-500 cursor-pointer"
              />
              <span className="text-[10px] text-zinc-600">Filters out shaky or uncertain signs</span>
            </div>

            {/* Stable Frames Required */}
            <div className="flex flex-col gap-1">
              <div className="flex justify-between text-zinc-400">
                <span>Stable Hold Frames:</span>
                <span className="font-mono text-indigo-400 font-medium">{stableFramesRequired} ({(stableFramesRequired * 0.25).toFixed(2)}s)</span>
              </div>
              <input
                type="range"
                min="2"
                max="6"
                step="1"
                value={stableFramesRequired}
                onChange={(e) => setStableFramesRequired(parseInt(e.target.value))}
                className="accent-indigo-500 cursor-pointer"
              />
              <span className="text-[10px] text-zinc-600">Consecutive frames required to confirm word</span>
            </div>

            {/* Word Boundary Pause Frames */}
            <div className="flex flex-col gap-1">
              <div className="flex justify-between text-zinc-400">
                <span>Boundary Pause:</span>
                <span className="font-mono text-indigo-400 font-medium">{pauseFramesRequired} ({(pauseFramesRequired * 0.25).toFixed(2)}s)</span>
              </div>
              <input
                type="range"
                min="1"
                max="5"
                step="1"
                value={pauseFramesRequired}
                onChange={(e) => setPauseFramesRequired(parseInt(e.target.value))}
                className="accent-indigo-500 cursor-pointer"
              />
              <span className="text-[10px] text-zinc-600">No-hand frames to reset duplicate lock</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
