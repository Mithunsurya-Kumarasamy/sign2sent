import React from 'react';
import { apiClient } from '../api/client';

export default function SentenceBuilder({ sequence, generatedSentence, onClear }) {
  const handleSpeak = async () => {
    if (!generatedSentence) return;
    try {
      await apiClient.post('/speak', { text: generatedSentence });
    } catch (err) {
      console.error("Speak failed", err);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Sequence Timeline */}
      <div className="p-5 rounded-2xl bg-slate-800/40 border border-slate-700/50 backdrop-blur-md">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">Sequence Timeline</h3>
        <div className="flex flex-wrap gap-2 min-h-[40px]">
          {sequence.length === 0 ? (
            <span className="text-sm text-slate-500">No signs detected yet.</span>
          ) : (
            sequence.map((sign, idx) => (
              <span key={idx} className="px-3 py-1 text-sm font-medium bg-slate-700/50 text-slate-200 rounded-lg border border-slate-600/50 animate-[fadeIn_0.3s_ease-out]">
                {sign}
              </span>
            ))
          )}
        </div>
      </div>

      {/* Finalized Sentence Output */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-indigo-900/40 to-blue-900/40 border border-indigo-500/30 shadow-[0_0_30px_rgba(79,70,229,0.15)] relative overflow-hidden group">
        <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-10 pointer-events-none mix-blend-overlay"></div>
        
        <div className="flex justify-between items-start mb-4 relative z-10">
          <h3 className="text-sm font-semibold text-indigo-300 uppercase tracking-widest">Translated Sentence</h3>
          <div className="flex gap-2">
            <button 
              onClick={handleSpeak}
              disabled={!generatedSentence}
              className="p-2 rounded-xl bg-indigo-500/20 text-indigo-300 hover:bg-indigo-500/40 hover:text-white transition-all disabled:opacity-30 disabled:cursor-not-allowed group-hover:shadow-[0_0_15px_rgba(99,102,241,0.5)]"
              title="Speak Aloud"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM14.657 2.929a1 1 0 011.414 0A9.972 9.972 0 0119 10a9.972 9.972 0 01-2.929 7.071 1 1 0 01-1.414-1.414A7.971 7.971 0 0017 10c0-2.21-.894-4.208-2.343-5.657a1 1 0 010-1.414zm-2.829 2.828a1 1 0 011.415 0A5.983 5.983 0 0115 10a5.984 5.984 0 01-1.757 4.243 1 1 0 01-1.415-1.415A3.984 3.984 0 0013 10a3.983 3.983 0 00-1.172-2.828 1 1 0 010-1.415z" clipRule="evenodd" />
              </svg>
            </button>
            <button 
              onClick={onClear}
              className="p-2 rounded-xl bg-slate-700/50 text-slate-400 hover:bg-slate-700 hover:text-white transition-all"
              title="Clear"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </button>
          </div>
        </div>

        <div className="min-h-[80px] flex items-center relative z-10">
          {generatedSentence ? (
            <p className="text-2xl md:text-3xl font-bold text-white tracking-wide leading-tight">
              {generatedSentence}
            </p>
          ) : (
            <p className="text-xl text-indigo-200/50 italic font-light">
              Start signing to generate a sentence...
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
