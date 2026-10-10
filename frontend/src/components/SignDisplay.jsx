import React from 'react';

export default function SignDisplay({ 
  currentSign, 
  confidence, 
  stabilityProgress = 0,
  isHandDetected = false,
  committedWord = null,
  confidenceThreshold = 0.70,
  stableFramesRequired = 3
}) {
  const isRecognized = currentSign && currentSign !== "None" && currentSign !== "Awaiting Gesture";
  const isLowConfidence = isRecognized && confidence < confidenceThreshold;
  const isCommitted = isRecognized && committedWord === currentSign;

  // Status message calculation
  let statusBadge = null;
  if (!isHandDetected) {
    statusBadge = (
      <span className="px-2.5 py-1 text-xs rounded-full font-medium bg-zinc-800 text-zinc-400 border border-zinc-700/50">
        No Hand Detected
      </span>
    );
  } else if (!isRecognized) {
    statusBadge = (
      <span className="px-2.5 py-1 text-xs rounded-full font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20 animate-pulse">
        Awaiting Stable Gesture
      </span>
    );
  } else if (isLowConfidence) {
    statusBadge = (
      <span className="px-2.5 py-1 text-xs rounded-full font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
        Low Confidence (&lt;{(confidenceThreshold * 100).toFixed(0)}%)
      </span>
    );
  } else if (isCommitted) {
    statusBadge = (
      <span className="px-2.5 py-1 text-xs rounded-full font-medium bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
        Word Added to Sequence
      </span>
    );
  } else {
    statusBadge = (
      <span className="px-2.5 py-1 text-xs rounded-full font-medium bg-indigo-500/15 text-indigo-400 border border-indigo-500/30 flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-ping"></span>
        Stabilizing... ({stabilityProgress}%)
      </span>
    );
  }

  return (
    <div className="bg-[#09090b] p-6 rounded-xl border border-zinc-800 shadow-sm flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-zinc-400">Current Frame Prediction</h2>
        {statusBadge}
      </div>
      
      <div className="flex flex-col items-center justify-center flex-1 min-h-[160px] bg-zinc-900/50 rounded-lg border border-zinc-800 p-6">
        {isRecognized ? (
          <div className="flex flex-col items-center w-full">
            <div className="text-5xl font-bold text-white tracking-tight mb-5 drop-shadow-sm flex items-center gap-3">
              <span>{currentSign}</span>
              {isCommitted && (
                <span className="text-xs px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                  LOCKED
                </span>
              )}
            </div>
            
            <div className="w-full max-w-[260px] flex flex-col gap-3">
              {/* Prediction Confidence Bar */}
              <div className="flex flex-col gap-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-zinc-500 font-medium">Model Confidence</span>
                  <span className={`font-mono font-medium ${confidence >= confidenceThreshold ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {(confidence * 100).toFixed(0)}% <span className="text-zinc-600 font-normal">(req. {(confidenceThreshold * 100).toFixed(0)}%)</span>
                  </span>
                </div>
                <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-200 ease-out ${
                      confidence >= confidenceThreshold ? 'bg-indigo-500' : 'bg-amber-500'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(0, confidence * 100))}%` }}
                  ></div>
                </div>
              </div>

              {/* Stability / Hold Progress Bar */}
              <div className="flex flex-col gap-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-zinc-500 font-medium">Hold Stability</span>
                  <span className={`font-mono font-medium ${stabilityProgress >= 100 ? 'text-emerald-400' : 'text-indigo-400'}`}>
                    {stabilityProgress}% <span className="text-zinc-600 font-normal">({Math.round((stabilityProgress / 100) * stableFramesRequired)}/{stableFramesRequired} frames)</span>
                  </span>
                </div>
                <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-150 ease-out ${
                      stabilityProgress >= 100 ? 'bg-emerald-500' : 'bg-indigo-400'
                    }`}
                    style={{ width: `${stabilityProgress}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-zinc-600 text-sm font-medium flex flex-col items-center gap-2">
            <svg className="w-8 h-8 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
            </svg>
            <p>Awaiting Sign Gesture...</p>
            <span className="text-xs text-zinc-600 font-normal">Hold a sign steadily in front of the camera</span>
          </div>
        )}
      </div>
    </div>
  );
}
