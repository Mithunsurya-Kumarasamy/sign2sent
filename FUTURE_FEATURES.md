# Future Features Roadmap

This document outlines highly technical, advanced features that can be added to **Sign2Sent** to evolve it into a state-of-the-art, production-ready system.

## 1. LLM-Powered NLP Engine 🧠
Currently, the `SentenceGenerator` relies on hardcoded grammar templates. 
* **Enhancement:** Integrate a local open-source LLM (e.g., Llama-3 via Ollama) or the Google Gemini API. Instead of strict template matching, the raw sequence of recognized signs (e.g., `["I", "HOSPITAL", "GO", "NOW"]`) is passed to the LLM to dynamically generate perfectly natural, context-aware English sentences.

## 2. Holistic Tracking (Face + Body) 👤
Sign language relies heavily on non-manual features (NMF) like facial expressions and body posture (e.g., raised eyebrows indicate a question).
* **Enhancement:** Upgrade from `mediapipe.solutions.hands` to `mediapipe.solutions.holistic` to capture 543 landmarks (hands, face mesh, and pose). Feeding this rich spatial data into a Transformer architecture would massively improve tonal and contextual accuracy.

## 3. Dynamic Custom Vocabulary (Few-Shot Learning) 🎯
Allowing users to customize the model without retraining the entire neural network.
* **Enhancement:** Add a "Train Custom Sign" interface in the React frontend. The user records a 5-second gesture, and the backend utilizes Siamese Networks or K-Nearest Neighbors (KNN) on the extracted CNN features to instantly begin recognizing the new sign on-the-fly.

## 4. Real-Time Analytics Dashboard 📊
Providing deep insights into the ML pipeline's performance.
* **Enhancement:** Integrate data visualization libraries like `Recharts` into a new dashboard tab. This would graph real-time metrics such as inference latency (ms/frame), model confidence thresholds over time, and usage heatmaps for the recognized vocabulary.

## 5. Dockerization & Cloud Deployment 🐳
To transition the system from local development to a globally accessible platform.
* **Enhancement:** Write a `Dockerfile` and `docker-compose.yml` that wraps the PyTorch backend, FastAPI, and Vite frontend into isolated, reproducible containers. Deploy the live demo to scalable cloud infrastructure like AWS, GCP, or Render.
