import argparse
import sys
import os

# Add the project root to sys.path so 'scripts' and 'ml' can be resolved
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.train_cnn import train_cnn
from scripts.train_lstm import train_lstm

def main():
    parser = argparse.ArgumentParser(description="Sign2Sent Training Pipeline")
    parser.add_argument("--stage", choices=["cnn", "lstm", "all"], default="all", help="Which stage to train")
    args = parser.parse_args()
    
    if args.stage in ["cnn", "all"]:
        print("="*40)
        print("STAGE 1: Training CNN Feature Extractor")
        print("="*40)
        train_cnn()
        
    if args.stage in ["lstm", "all"]:
        print("="*40)
        print("STAGE 2: Training LSTM Sequence Recognizer")
        print("="*40)
        train_lstm()

if __name__ == "__main__":
    main()
