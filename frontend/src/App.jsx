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
      console.error("Failed to update sentence", err);
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
    <div className="min-h-screen relative overflow-hidden bg-brand-dark">
      {/* Animated Ambient Background Blobs */}
      <div className="absolute top-0 -left-4 w-96 h-96 bg-brand-primary rounded-full mix-blend-screen filter blur-[128px] opacity-20 animate-blob"></div>
      <div className="absolute top-0 -right-4 w-96 h-96 bg-brand-secondary rounded-full mix-blend-screen filter blur-[128px] opacity-20 animate-blob animation-delay-2000"></div>
      <div className="absolute -bottom-32 left-1/2 w-96 h-96 bg-brand-accent rounded-full mix-blend-screen filter blur-[128px] opacity-20 animate-blob animation-delay-4000"></div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 flex flex-col gap-10">
        
        {/* Header */}
        <header className="flex flex-col md:flex-row items-center justify-between gap-4 animate-fade-in">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-brand-primary to-brand-secondary flex items-center justify-center shadow-lg shadow-brand-primary/30">
              <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
              </svg>
            </div>
            <div>
              <h1 className="text-4xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400">
                Sign2Sent AI
              </h1>
              <p className="text-sm text-brand-secondary tracking-widest font-medium uppercase mt-1">Real-Time Translation</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3 px-5 py-2.5 rounded-full glass-card hover:bg-slate-800/60 transition-colors">
            <div className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
            </div>
            <span className="text-sm font-semibold text-slate-200">System Online</span>
          </div>
        </header>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column (Camera + Controls) */}
          <div className="lg:col-span-7 flex flex-col gap-6 animate-slide-up" style={{ animationDelay: '0.1s' }}>
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
          
          {/* Right Column (NLP & Analysis) */}
          <div className="lg:col-span-5 flex flex-col gap-6 animate-slide-up" style={{ animationDelay: '0.2s' }}>
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
      </div>
    </div>
  );
}
