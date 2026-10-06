import React from 'react';
import { apiClient } from '../api/client';

export default function SentenceBuilder({ sequence, generatedSentence, onClear }) {
  const handleSpeak = async () => {
    if (!generatedSentence) return;
    try {
      await apiClient.post('/speak', { text: generatedSentence });
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Sequence Timeline */}
      <div className="glass-card p-6 rounded-3xl">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-[0.2em] mb-4">Sequence Timeline</h3>
        <div className="flex flex-wrap gap-2.5 min-h-[44px]">
          {sequence.length === 0 ? (
            <span className="text-sm font-medium text-slate-500 italic flex items-center h-full">Listening for signs...</span>
          ) : (
            sequence.map((sign, idx) => (
              <span key={idx} className="px-4 py-1.5 text-sm font-bold bg-slate-800 text-slate-200 rounded-xl border border-slate-700 shadow-sm animate-fade-in flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-primary"></span>
                {sign}
              </span>
            ))
          )}
        </div>
      </div>

      {/* NLP Generated Sentence */}
      <div className="p-8 rounded-3xl bg-gradient-to-br from-brand-primary/20 via-brand-accent/10 to-transparent border border-brand-primary/30 shadow-[0_0_40px_rgba(79,70,229,0.1)] relative overflow-hidden group">
        <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-5 pointer-events-none mix-blend-overlay"></div>
        
        <div className="flex justify-between items-start mb-6 relative z-10">
          <h3 className="text-xs font-bold text-brand-secondary uppercase tracking-[0.2em]">Translated Output</h3>
          <div className="flex gap-3">
            <button 
              onClick={handleSpeak}
              disabled={!generatedSentence}
              className="p-2.5 rounded-xl bg-brand-primary text-white shadow-lg shadow-brand-primary/40 hover:bg-indigo-500 hover:shadow-indigo-500/50 transition-all disabled:opacity-30 disabled:cursor-not-allowed hover:-translate-y-0.5 active:translate-y-0"
              title="Speak Aloud"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
              </svg>
            </button>
            <button 
              onClick={onClear}
              className="p-2.5 rounded-xl bg-slate-800 text-slate-400 border border-slate-700 hover:bg-slate-700 hover:text-white transition-all hover:-translate-y-0.5 active:translate-y-0"
              title="Clear Sequence"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>
        </div>

        <div className="min-h-[100px] flex items-center relative z-10">
          {generatedSentence ? (
            <p className="text-3xl md:text-4xl font-extrabold text-white tracking-tight leading-tight drop-shadow-md">
              {generatedSentence}
            </p>
          ) : (
            <p className="text-2xl text-slate-400/50 font-medium">
              Start signing...
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
