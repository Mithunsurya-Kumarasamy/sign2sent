import React, { useState } from 'react';
import WebcamView from './components/WebcamView';
import SignDisplay from './components/SignDisplay';
import SentenceBuilder from './components/SentenceBuilder';
import ControlPanel from './components/ControlPanel';
import { apiClient } from './api/client';

export default function App() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentSign, setCurrentSign] = useState('None');
  const [confidence, setConfidence] = useState(0.0);
  const [sequence, setSequence] = useState([]);
  const [generatedSentence, setGeneratedSentence] = useState('');

  const handlePrediction = async (data) => {
    if (data && data.sign) {
      setCurrentSign(data.sign);
      setConfidence(data.confidence);
      
      setSequence(prev => {
        const newSeq = [...prev, data.sign];
        updateSentence(newSeq);
        return newSeq;
      });
    }
  };

  const updateSentence = async (currentSequence) => {
    try {
      const res = await apiClient.post('/sentence', { sequence: currentSequence });
      if (res.sentence) setGeneratedSentence(res.sentence);
    } catch (err) {
      console.error(err);
    }
  };

  const handleClear = () => {
    setSequence([]);
    setGeneratedSentence('');
    setCurrentSign('None');
    setConfidence(0.0);
  };

  const handleForceFinalize = async () => {
    setIsProcessing(false);
    await updateSentence(sequence);
    setSequence([]);
  };

  return (
    <div className="min-h-screen bg-[#09090b] text-zinc-100 font-sans selection:bg-indigo-500/30">
      
      {/* Sleek Top Navigation */}
      <nav className="bg-[#09090b] border-b border-zinc-800 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-indigo-500 flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
              </svg>
            </div>
            <h1 className="text-xl font-bold tracking-tight text-zinc-100">
              Sign2Sent
            </h1>
          </div>
          
          <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-emerald-950/30 border border-emerald-900/50">
            <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
            <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">System Online</span>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col gap-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
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
            />
          </div>
          
          <div className="lg:col-span-5 flex flex-col gap-6">
            <SignDisplay 
              currentSign={currentSign}
              confidence={confidence}
            />
            <SentenceBuilder 
              sequence={sequence}
              generatedSentence={generatedSentence}
              onClear={handleClear}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
