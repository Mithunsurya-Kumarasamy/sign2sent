import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from ml.utils.config import CNN_INPUT_SIZE

def get_base_transform():
    """
    Base transforms for validation/testing: Resize and convert to tensor.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(CNN_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_train_transform():
    """
    Training transforms with augmentation.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(CNN_INPUT_SIZE),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
def preprocess_frame(frame, transform):
    """
    Preprocess a single frame.
    """
    return transform(frame)
