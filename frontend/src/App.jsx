import React, { useState, useRef, useCallback } from 'react';
import WebcamView from './components/WebcamView';
import SignDisplay from './components/SignDisplay';
import SentenceBuilder from './components/SentenceBuilder';
import ControlPanel from './components/ControlPanel';
import { apiClient } from './api/client';

export default function App() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentSign, setCurrentSign] = useState('None');
  const [confidence, setConfidence] = useState(0.0);
  const [stabilityProgress, setStabilityProgress] = useState(0);
  const [isHandDetected, setIsHandDetected] = useState(false);
  const [committedWord, setCommittedWord] = useState(null);
  const [sequence, setSequence] = useState([]);
  const [generatedSentence, setGeneratedSentence] = useState('');

  // Configurable debouncing parameters
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.70);
  const [stableFramesRequired, setStableFramesRequired] = useState(3); // 3 frames @ 250ms = ~750ms
  const [pauseFramesRequired, setPauseFramesRequired] = useState(2);    // 2 frames @ 250ms = ~500ms

  // High-frequency tracker refs (immune to React state closure latency)
  const candidateRef = useRef({ sign: null, count: 0 });
  const committedWordRef = useRef(null);
  const pauseCountRef = useRef(0);

  // Calls the backend /api/sentence endpoint to generate grammatically correct sentence
  const updateSentence = useCallback(async (currentSequence) => {
    if (!currentSequence || currentSequence.length === 0) {
      setGeneratedSentence('');
      return;
    }
    try {
      const res = await apiClient.post('/sentence', { sequence: currentSequence });
      if (res && res.sentence !== undefined) {
        setGeneratedSentence(res.sentence);
      }
    } catch (err) {
      console.error('Failed to generate sentence:', err);
    }
  }, []);

  // Frame prediction handler from WebcamView
  const handlePrediction = useCallback((data) => {
    if (!data) return;

    const handDetected = Boolean(data.hand_detected);
    const sign = data.sign;
    const conf = typeof data.confidence === 'number' ? data.confidence : 0.0;

    // 1. Check Hand Detection (Requirement 7: If no hand detected, do not add anything)
    if (!handDetected) {
      setIsHandDetected(false);
      setCurrentSign('None');
      setConfidence(0.0);
      setStabilityProgress(0);

      // Track pause duration
      pauseCountRef.current += 1;
      candidateRef.current = { sign: null, count: 0 };

      // Requirement 6: Word boundary detection via short pause/no-hand period
      // When pause reaches threshold, clear the committed word lock to allow signing the same word again
      if (pauseCountRef.current >= pauseFramesRequired) {
        committedWordRef.current = null;
        setCommittedWord(null);
      }
      return;
    }

    // Hand is present
    setIsHandDetected(true);
    pauseCountRef.current = 0; // reset pause counter

    // Hand present, but no sign detected yet (e.g. buffer filling or gesture in transition)
    if (!sign) {
      setCurrentSign('Awaiting Gesture');
      setConfidence(conf);
      setStabilityProgress(0);
      candidateRef.current = { sign: null, count: 0 };
      return;
    }

    // 2. Confidence Threshold Check (Requirement 4: Low confidence discarded)
    if (conf < confidenceThreshold) {
      setCurrentSign(sign);
      setConfidence(conf);
      // Discard candidate count on noisy / low-confidence frames
      candidateRef.current = { sign: null, count: 0 };
      setStabilityProgress(0);
      return;
    }

    // Valid high-confidence sign frame
    setCurrentSign(sign);
    setConfidence(conf);

    // 3. Debouncing & Stability Tracking (Requirement 3 & 5)
    if (candidateRef.current.sign === sign) {
      // Same candidate sign maintained
      candidateRef.current.count += 1;
    } else {
      // New candidate sign started
      candidateRef.current = { sign, count: 1 };
    }

    const currentCount = candidateRef.current.count;
    const progress = Math.min(100, Math.round((currentCount / stableFramesRequired) * 100));
    setStabilityProgress(progress);

    // 4. Stable Word Confirmation
    if (currentCount >= stableFramesRequired) {
      // The sign has been consistently held for the required number of frames
      // Prevent repeated frame predictions from repeating the word while held (Requirement 5)
      if (committedWordRef.current !== sign) {
        // Success: Confirm and append the new word!
        committedWordRef.current = sign;
        setCommittedWord(sign);

        setSequence(prev => {
          const newSeq = [...prev, sign];
          updateSentence(newSeq);
          return newSeq;
        });
      }
    }
  }, [confidenceThreshold, stableFramesRequired, pauseFramesRequired, updateSentence]);

  // Full Clear / Reset (Requirement 12)
  const handleClear = () => {
    setSequence([]);
    setGeneratedSentence('');
    setCurrentSign('None');
    setConfidence(0.0);
    setStabilityProgress(0);
    setIsHandDetected(false);
    setCommittedWord(null);
    candidateRef.current = { sign: null, count: 0 };
    committedWordRef.current = null;
    pauseCountRef.current = 0;
  };

  // Undo the most recent word in the sequence
  const handleRemoveLastWord = () => {
    setSequence(prev => {
      if (prev.length === 0) return prev;
      const nextSeq = prev.slice(0, -1);
      updateSentence(nextSeq);
      return nextSeq;
    });
    committedWordRef.current = null;
    setCommittedWord(null);
  };

  // Manual addition of word (useful for demonstration and viva testing)
  const handleManualAddWord = (word) => {
    if (!word) return;
    setSequence(prev => {
      const nextSeq = [...prev, word];
      updateSentence(nextSeq);
      return nextSeq;
    });
    committedWordRef.current = word;
    setCommittedWord(word);
  };

  // Load preset sequence directly (e.g. HELLO -> MY -> NAME -> IS -> JOHN)
  const handleLoadPreset = (presetWords) => {
    setSequence(presetWords);
    updateSentence(presetWords);
    if (presetWords.length > 0) {
      const lastWord = presetWords[presetWords.length - 1];
      committedWordRef.current = lastWord;
      setCommittedWord(lastWord);
    }
  };

  // Finalize sentence (Requirement 13)
  const handleForceFinalize = async () => {
    setIsProcessing(false);
    await updateSentence(sequence);
  };

  return (
    <div className="min-h-screen bg-[#09090b] text-zinc-100 font-sans selection:bg-indigo-500/30">

      {/* Top Navigation */}
      <nav className="bg-[#09090b] border-b border-zinc-800 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-indigo-500 flex items-center justify-center shadow-xs">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-zinc-100">
                Sign2Sent
              </h1>
              <span className="text-[10px] text-zinc-500 hidden sm:inline">Continuous Sign Language Translation System</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-emerald-950/30 border border-emerald-900/50">
              <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
              <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">System Online</span>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Video & Control Actions */}
          <div className="lg:col-span-7 flex flex-col gap-6">
            <WebcamView
              isProcessing={isProcessing}
              setIsProcessing={setIsProcessing}
              onPrediction={handlePrediction}
            />
            <ControlPanel
              isProcessing={isProcessing}
              setIsProcessing={setIsProcessing}
              forceFinalize={handleForceFinalize}
              onClear={handleClear}
              confidenceThreshold={confidenceThreshold}
              setConfidenceThreshold={setConfidenceThreshold}
              stableFramesRequired={stableFramesRequired}
              setStableFramesRequired={setStableFramesRequired}
              pauseFramesRequired={pauseFramesRequired}
              setPauseFramesRequired={setPauseFramesRequired}
            />
          </div>

          {/* Right Column: Sign Recognition & Sentence Generation */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            <SignDisplay
              currentSign={currentSign}
              confidence={confidence}
              stabilityProgress={stabilityProgress}
              isHandDetected={isHandDetected}
              committedWord={committedWord}
              confidenceThreshold={confidenceThreshold}
              stableFramesRequired={stableFramesRequired}
            />
            <SentenceBuilder
              sequence={sequence}
              generatedSentence={generatedSentence}
              onClear={handleClear}
              onRemoveLastWord={handleRemoveLastWord}
              onAddWord={handleManualAddWord}
              onLoadPreset={handleLoadPreset}
            />
          </div>
        </div>

        {/* Academic Architecture Pipeline Overview */}
        <div className="p-4 rounded-xl border border-zinc-800/80 bg-zinc-900/30 flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-zinc-400">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-semibold text-zinc-300">Architecture Pipeline:</span>
            <span className="px-2 py-0.5 rounded bg-zinc-800 text-zinc-300">Webcam Frame (4 FPS)</span>
            <span>→</span>
            <span className="px-2 py-0.5 rounded bg-zinc-800 text-zinc-300">MediaPipe Hand Crop</span>
            <span>→</span>
            <span className="px-2 py-0.5 rounded bg-zinc-800 text-zinc-300">CNN+LSTM Sign Classifier</span>
            <span>→</span>
            <span className="px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">Stability Filter ({stableFramesRequired} frames)</span>
            <span>→</span>
            <span className="px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">Duplicate Suppressor</span>
            <span>→</span>
            <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">Sequence Accumulator</span>
            <span>→</span>
            <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">NLP Sentence Generator</span>
          </div>
        </div>
      </main>
    </div>
  );
}
