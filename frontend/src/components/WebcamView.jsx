import React, { useRef, useEffect, useState, useCallback } from 'react';
import { apiClient } from '../api/client';

export default function WebcamView({ onPrediction, isProcessing, setIsProcessing }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [hasCamera, setHasCamera] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    async function setupCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setHasCamera(true);
        }
      } catch (err) {
        setError('Camera permission denied or camera not found.');
      }
    }
    setupCamera();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const captureFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !isProcessing) return;

    const ctx = canvasRef.current.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, 640, 480);
    
    canvasRef.current.toBlob(async (blob) => {
      if (!blob) return;
      const formData = new FormData();
      formData.append('file', blob, 'frame.jpg');

      try {
        const data = await apiClient.postForm('/predict', formData);
        onPrediction(data);
      } catch (err) {
        console.error(err);
      }
    }, 'image/jpeg', 0.8);
  }, [isProcessing, onPrediction]);

  useEffect(() => {
    let interval;
    if (isProcessing && hasCamera) {
      interval = setInterval(captureFrame, 250); 
    }
    return () => clearInterval(interval);
  }, [isProcessing, hasCamera, captureFrame]);

  return (
    <div className="glass-panel rounded-3xl overflow-hidden relative group aspect-[4/3] flex flex-col justify-center">
      {error ? (
        <div className="text-red-400 p-8 text-center m-auto">
          <svg className="w-12 h-12 mx-auto mb-4 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          <p className="font-medium">{error}</p>
        </div>
      ) : (
        <>
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted 
            className={`w-full h-full object-cover scale-x-[-1] transition-all duration-700 ${!isProcessing && 'grayscale brightness-50 contrast-125'}`}
          />
          <canvas ref={canvasRef} width="640" height="480" className="hidden" />
          
          {/* HUD Overlay */}
          <div className="absolute inset-0 pointer-events-none box-border border-[6px] border-transparent group-hover:border-white/5 transition-all duration-500 rounded-3xl"></div>
          
          <div className="absolute top-6 left-6 flex gap-3">
            <div className={`px-4 py-1.5 rounded-full backdrop-blur-md border shadow-lg flex items-center gap-2 transition-all ${isProcessing ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-300' : 'bg-slate-800/60 border-slate-600/50 text-slate-400'}`}>
              <div className={`w-2.5 h-2.5 rounded-full ${isProcessing ? 'bg-emerald-400 animate-pulse shadow-[0_0_10px_#34d399]' : 'bg-slate-500'}`}></div>
              <span className="text-xs font-bold uppercase tracking-wider">{isProcessing ? 'Live Feed' : 'Standby'}</span>
            </div>
          </div>
          
          <div className="absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-slate-950 via-slate-900/60 to-transparent pointer-events-none"></div>
        </>
      )}
    </div>
  );
}
