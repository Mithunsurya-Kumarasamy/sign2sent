import React, { useState } from 'react';
import { apiClient } from '../api/client';

export default function SentenceBuilder({ 
  sequence = [], 
  generatedSentence = '', 
  onClear,
  onRemoveLastWord,
  onAddWord,
  onLoadPreset
}) {
  const [customWord, setCustomWord] = useState('');
  const [isCopied, setIsCopied] = useState(false);

  const handleSpeak = async () => {
    if (!generatedSentence) return;
    try {
      await apiClient.post('/speak', { text: generatedSentence });
    } catch (err) {
      console.error(err);
    }
  };

  const handleCopy = () => {
    if (!generatedSentence) return;
    navigator.clipboard.writeText(generatedSentence);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  const handleAddSubmit = (e) => {
    e.preventDefault();
    if (!customWord.trim()) return;
    if (onAddWord) {
      onAddWord(customWord.trim().toUpperCase());
      setCustomWord('');
    }
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Recognized Words Timeline */}
      <div className="bg-[#09090b] p-6 rounded-xl border border-zinc-800 shadow-sm">
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-zinc-400">Recognized Words</h3>
            <span className="px-2 py-0.5 rounded text-xs font-mono font-medium bg-zinc-800 text-indigo-400 border border-zinc-700">
              {sequence.length}
            </span>
          </div>
          
          <div className="flex items-center gap-2">
            {sequence.length > 0 && onRemoveLastWord && (
              <button 
                onClick={onRemoveLastWord}
                className="text-xs px-2.5 py-1 rounded bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800 hover:border-zinc-700 transition-colors flex items-center gap-1"
                title="Undo last word"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a5 5 0 015 5v2m0 0l-4-4m4 4l4-4" />
                </svg>
                Undo
              </button>
            )}
            
            {sequence.length > 0 && (
              <button 
                onClick={onClear}
                className="text-xs px-2.5 py-1 rounded bg-rose-500/10 text-rose-400 hover:bg-rose-500/20 border border-rose-500/30 transition-colors flex items-center gap-1"
                title="Clear sequence"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                Clear
              </button>
            )}
          </div>
        </div>

        {/* Word Sequence Badges */}
        <div className="flex flex-wrap gap-2 min-h-[48px] p-2.5 bg-zinc-900/40 rounded-lg border border-zinc-800/80 items-center">
          {sequence.length === 0 ? (
            <span className="text-sm text-zinc-600 italic px-2">
              No signs accumulated yet. Hold signs stably to add words...
            </span>
          ) : (
            sequence.map((sign, idx) => (
              <span 
                key={idx} 
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-semibold bg-zinc-800/90 text-indigo-300 rounded-md border border-indigo-500/30 shadow-xs transition-transform transform active:scale-95"
              >
                <span className="text-[10px] font-mono font-medium text-zinc-500">#{idx + 1}</span>
                <span>{sign}</span>
              </span>
            ))
          )}
        </div>

        {/* Quick Demo Presets & Manual Test Input (Ideal for Viva / Evaluation) */}
        <div className="mt-4 pt-4 border-t border-zinc-800/70 flex flex-col gap-2.5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-xs font-medium text-zinc-500">Viva Test Presets:</span>
            <div className="flex flex-wrap gap-1.5">
              <button
                type="button"
                onClick={() => onLoadPreset && onLoadPreset(["HELLO", "MY", "NAME", "IS", "JOHN"])}
                className="text-xs px-2.5 py-1 rounded bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-500/25 transition-colors font-medium"
              >
                HELLO → MY → NAME → IS → JOHN
              </button>
              <button
                type="button"
                onClick={() => onLoadPreset && onLoadPreset(["I", "NEED", "WATER"])}
                className="text-xs px-2 py-1 rounded bg-zinc-800 text-zinc-300 border border-zinc-700 hover:bg-zinc-700 transition-colors"
              >
                I → NEED → WATER
              </button>
              <button
                type="button"
                onClick={() => onLoadPreset && onLoadPreset(["HELLO", "HOW", "YOU"])}
                className="text-xs px-2 py-1 rounded bg-zinc-800 text-zinc-300 border border-zinc-700 hover:bg-zinc-700 transition-colors"
              >
                HELLO → HOW → YOU
              </button>
            </div>
          </div>

          {/* Manual Word Add Input */}
          <form onSubmit={handleAddSubmit} className="flex gap-2 items-center mt-1">
            <input
              type="text"
              value={customWord}
              onChange={(e) => setCustomWord(e.target.value)}
              placeholder="Or test word (e.g. HELLO, WATER, JOHN)..."
              className="flex-1 text-xs px-3 py-1.5 bg-zinc-900 border border-zinc-700 rounded text-zinc-200 placeholder-zinc-600 focus:outline-hidden focus:border-indigo-500"
            />
            <button
              type="submit"
              disabled={!customWord.trim()}
              className="text-xs px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40 text-zinc-200 border border-zinc-700 rounded font-medium transition-colors cursor-pointer"
            >
              + Add Word
            </button>
          </form>
        </div>
      </div>

      {/* Translated English Sentence */}
      <div className="bg-[#09090b] p-6 rounded-xl border border-zinc-800 shadow-sm flex flex-col">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h3 className="text-sm font-semibold text-zinc-400">Generated English Sentence</h3>
            <p className="text-xs text-zinc-600">Grammar & template synthesis from accumulated sequence</p>
          </div>
          
          <div className="flex gap-2">
            <button 
              onClick={handleCopy}
              disabled={!generatedSentence}
              className="p-1.5 rounded bg-zinc-900 text-zinc-400 border border-zinc-700 hover:bg-zinc-800 hover:text-zinc-200 transition-colors disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
              title="Copy to clipboard"
            >
              {isCopied ? (
                <svg className="h-5 w-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              )}
            </button>

            <button 
              onClick={handleSpeak}
              disabled={!generatedSentence}
              className="p-1.5 rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 hover:bg-indigo-500/30 transition-colors disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
              title="Speak Aloud (TTS)"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
              </svg>
            </button>
          </div>
        </div>

        <div className="min-h-[85px] flex items-center bg-zinc-900/50 rounded-lg p-5 border border-zinc-800/80">
          {generatedSentence ? (
            <p className="text-xl md:text-2xl font-semibold text-emerald-400 tracking-tight leading-snug">
              {generatedSentence}
            </p>
          ) : (
            <p className="text-base text-zinc-600 italic">
              Translated English sentence will appear here as signs are accumulated...
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
