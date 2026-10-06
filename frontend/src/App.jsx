import React, { useState, useEffect } from 'react';
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
      
      // Update sequence if it's a new confident sign
      setSequence(prev => {
        const newSeq = [...prev, data.sign];
        updateSentence(newSeq);
        return newSeq;
      });
    } else if (data && !data.hand_detected) {
      // Optional: Dim confidence or set status if no hand is detected
    }
  };

  const updateSentence = async (currentSequence) => {
    try {
      const res = await apiClient.post('/sentence', { sequence: currentSequence });
      if (res.sentence) {
        setGeneratedSentence(res.sentence);
      }
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
    setIsProcessing(false); // Pause when forcing finalize
    await updateSentence(sequence);
    setSequence([]);
  };

  return (
    <div className="min-h-screen bg-brand-dark text-slate-200 font-sans selection:bg-brand-primary/30">
      {/* Background Decor */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px]"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-indigo-600/10 rounded-full blur-[120px]"></div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-8 relative z-10 flex flex-col gap-8">
        
        {/* Header */}
        <header className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl md:text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-400 to-blue-500 tracking-tight mb-2">
              Sign2Sent
            </h1>
            <p className="text-slate-400 font-medium">Real-Time Sign Language Translation</p>
          </div>
          <div className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-full bg-slate-800/50 border border-slate-700/50">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
            <span className="text-sm font-semibold text-slate-300">System Ready</span>
          </div>
        </header>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Camera & Controls */}
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
          
          {/* Right Column: Insights & NLP */}
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
      </div>
    </div>
  );
}
