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
        console.error("Camera error:", err);
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
        console.error("Prediction failed:", err);
      }
    }, 'image/jpeg', 0.8);
  }, [isProcessing, onPrediction]);

  // Inference loop
  useEffect(() => {
    let interval;
    if (isProcessing && hasCamera) {
      interval = setInterval(captureFrame, 250); // 4 frames per second
    }
    return () => clearInterval(interval);
  }, [isProcessing, hasCamera, captureFrame]);

  return (
    <div className="relative overflow-hidden rounded-2xl shadow-[0_0_40px_rgba(59,130,246,0.3)] bg-slate-900 border border-slate-700/50 backdrop-blur-xl">
      {error ? (
        <div className="flex items-center justify-center h-[480px] text-red-400 p-8 text-center bg-slate-800/50">
          {error}
        </div>
      ) : (
        <>
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted 
            className="w-full h-auto max-h-[500px] object-cover scale-x-[-1]"
          />
          <canvas ref={canvasRef} width="640" height="480" className="hidden" />
          
          <div className="absolute top-4 left-4 flex gap-2">
            <span className={`px-3 py-1 text-xs font-semibold uppercase tracking-wider rounded-full flex items-center gap-2 backdrop-blur-md ${isProcessing ? 'bg-green-500/20 text-green-300 border border-green-500/30' : 'bg-red-500/20 text-red-300 border border-red-500/30'}`}>
              <div className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-green-400 animate-pulse' : 'bg-red-500'}`}></div>
              {isProcessing ? 'Live Inference' : 'Paused'}
            </span>
          </div>
          
          <div className="absolute inset-0 bg-gradient-to-t from-brand-dark/80 via-transparent to-transparent pointer-events-none"></div>
        </>
      )}
    </div>
  );
}
