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
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
def preprocess_frame(frame, transform):
    """
    Preprocess a single frame.
    """
    return transform(frame)
