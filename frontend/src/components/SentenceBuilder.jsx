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
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
        <h3 className="text-sm font-semibold text-slate-700 mb-4">Sequence Timeline</h3>
        <div className="flex flex-wrap gap-2 min-h-[40px]">
          {sequence.length === 0 ? (
            <span className="text-sm text-slate-400 italic flex items-center">Listening for sequence...</span>
          ) : (
            sequence.map((sign, idx) => (
              <span key={idx} className="px-3 py-1 text-sm font-medium bg-slate-100 text-slate-700 rounded-md border border-slate-200">
                {sign}
              </span>
            ))
          )}
        </div>
      </div>

      {/* NLP Generated Sentence */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col">
        <div className="flex justify-between items-start mb-6">
          <h3 className="text-sm font-semibold text-slate-700">Translated Sentence</h3>
          <div className="flex gap-2">
            <button 
              onClick={handleSpeak}
              disabled={!generatedSentence}
              className="p-1.5 rounded-md bg-indigo-50 text-indigo-600 hover:bg-indigo-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Speak Aloud"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
              </svg>
            </button>
            <button 
              onClick={onClear}
              className="p-1.5 rounded-md bg-slate-50 text-slate-500 border border-slate-200 hover:bg-slate-100 transition-colors"
              title="Clear Sequence"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>
        </div>

        <div className="min-h-[80px] flex items-center bg-slate-50 rounded-lg p-4 border border-slate-100">
          {generatedSentence ? (
            <p className="text-xl md:text-2xl font-bold text-slate-800 leading-tight">
              {generatedSentence}
            </p>
          ) : (
            <p className="text-lg text-slate-400 italic">
              Translated output will appear here...
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
