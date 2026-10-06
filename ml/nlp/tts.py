import pyttsx3
import threading

class TextToSpeech:
    def __init__(self):
        self._engine = None
        self._lock = threading.Lock()
        self._initialize_engine()
        
    def _initialize_engine(self):
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', 150)    # Speed of speech
            self._engine.setProperty('volume', 1.0)  # Volume 0-1
        except Exception as e:
            print(f"TTS Initialization Error: {e}")
            self._engine = None

    def speak(self, text: str):
        """
        Speaks the given text in a separate thread to avoid blocking the main application.
        """
        if not text or self._engine is None:
            return
            
        def _speak_thread():
            with self._lock:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as e:
                    print(f"TTS Execution Error: {e}")
                    
        # Run in a daemon thread so it doesn't block program exit
        t = threading.Thread(target=_speak_thread, daemon=True)
        t.start()
