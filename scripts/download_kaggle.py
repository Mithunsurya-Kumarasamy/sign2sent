import os
import kagglehub

def main():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("mishbhaul/wlasl-1000-preprocessed")
    print("\nDataset successfully downloaded!")
    print("Path to dataset files:", path)
    
    print("\nContents of the directory:")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

if __name__ == "__main__":
    main()
