# Sign2Sent

## Overview
**Sign2Sent — Real-Time Sign Language to Sentence Translation**
Sign2Sent is an academic prototype system designed to translate real-time sign language gestures into meaningful English sentences using computer vision and deep learning. 

**Important Note:** Sign2Sent currently supports a predefined vocabulary and sentence patterns and is intended as an academic prototype rather than unrestricted sign-language translation.

## Problem Statement
Bridging the communication gap between the hearing impaired and the general public is a significant challenge. Sign language is rich and dynamic, making real-time translation complex.

## Objective
To build a modular, real-time sign language recognition system that detects hand gestures, translates sequential signs, and generates coherent sentences using predefined templates.

## Features
- Real-time Hand Tracking using MediaPipe.
- Spatial Feature Extraction using a Convolutional Neural Network (CNN).
- Temporal Sequence Recognition using Long Short-Term Memory (LSTM) networks.
- Sentence Formation from predefined vocabulary and grammar rules.
- Optional Text-to-Speech (TTS) integration.
- FastAPI Backend for inference and model management.
- React Frontend with live webcam integration.

## System Architecture
WEBCAM -> VIDEO FRAMES -> MEDIAPIPE HAND DETECTION -> PREPROCESSING -> CNN -> SPATIAL FEATURES -> LSTM -> TEMPORAL SEQUENCE RECOGNITION -> SIGN SEQUENCE -> SENTENCE GENERATION -> ENGLISH SENTENCE -> OPTIONAL TEXT-TO-SPEECH

## Technology Stack
- **Machine Learning**: Python, PyTorch, OpenCV, MediaPipe, NumPy, Scikit-learn
- **NLP**: Custom Template-based Sentence Generator (or NLTK)
- **Backend**: FastAPI, Uvicorn
- **Frontend**: React, Vite, Tailwind CSS

## Project Structure
(See directory structure for details on separation of concerns.)

## Limitations
- Supports a limited predefined vocabulary.
- Uses strict template-based sentence generation.
- Not a universal or unrestricted sign language translator.

## Future Improvements
- Larger vocabulary.
- More robust signer-independent models.
- Support for continuous sign language translation.
- Transformer-based temporal models.
- Mobile/Edge deployment.

## Installation & Usage
(To be added in future phases)

## Team
- Mithunsurya Kumarasamy
