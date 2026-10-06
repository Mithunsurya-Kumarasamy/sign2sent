import React, { useRef, useEffect, useState, useCallback } from 'react';
import { apiClient } from '../api/client';

export default function WebcamView({ onPrediction, isProcessing, setIsProcessing }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [hasCamera, setHasCamera] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(true);
  const [error, setError] = useState('');

  const setupCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setHasCamera(true);
        setIsCameraOn(true);
      }
    } catch (err) {
      setError('Camera permission denied or camera not found.');
      setHasCamera(false);
      setIsCameraOn(false);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraOn(false);
    setIsProcessing(false); // Can't process if camera is off
  }, [setIsProcessing]);

  useEffect(() => {
    setupCamera();
    return stopCamera;
  }, [setupCamera, stopCamera]);

  const toggleCamera = () => {
    if (isCameraOn) {
      stopCamera();
    } else {
      setupCamera();
    }
  };

  const captureFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !isProcessing || !isCameraOn) return;

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
  }, [isProcessing, isCameraOn, onPrediction]);

  useEffect(() => {
    let interval;
    if (isProcessing && hasCamera && isCameraOn) {
      interval = setInterval(captureFrame, 250); 
    }
    return () => clearInterval(interval);
  }, [isProcessing, hasCamera, isCameraOn, captureFrame]);

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col">
      <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between bg-slate-50">
        <h2 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
          <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          Video Feed
        </h2>
        
        <div className="flex items-center gap-3">
          {/* Status Indicator */}
          {isCameraOn && (
            <div className={`px-2 py-1 rounded text-xs font-medium flex items-center gap-1.5 ${isProcessing ? 'text-indigo-700 bg-indigo-50' : 'text-slate-500 bg-slate-200'}`}>
              <div className={`w-1.5 h-1.5 rounded-full ${isProcessing ? 'bg-indigo-500 animate-pulse' : 'bg-slate-400'}`}></div>
              {isProcessing ? 'Inference Active' : 'Idle'}
            </div>
          )}
          
          {/* Toggle Camera Button */}
          <button 
            onClick={toggleCamera}
            className={`text-xs font-medium px-3 py-1.5 rounded-md border transition-colors ${
              isCameraOn 
                ? 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50' 
                : 'bg-indigo-600 border-indigo-600 text-white hover:bg-indigo-700'
            }`}
          >
            {isCameraOn ? 'Turn Off Camera' : 'Turn On Camera'}
          </button>
        </div>
      </div>

      <div className="relative aspect-[4/3] bg-slate-100 flex flex-col items-center justify-center">
        {error && !isCameraOn ? (
          <div className="text-slate-500 text-sm">{error}</div>
        ) : !isCameraOn ? (
          <div className="text-slate-400 flex flex-col items-center gap-2">
            <svg className="w-10 h-10 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
            </svg>
            <p className="text-sm font-medium">Camera is disabled</p>
          </div>
        ) : (
          <>
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted 
              className={`w-full h-full object-cover scale-x-[-1] transition-opacity duration-300 ${!isProcessing && 'opacity-75'}`}
            />
            <canvas ref={canvasRef} width="640" height="480" className="hidden" />
          </>
        )}
      </div>
    </div>
  );
}
