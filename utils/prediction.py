import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np

class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']
trained_model = None

#Since we have saved the mode as state_dict in this case only model weights are stored but not the structure
#So we need declare the model structure as well 
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        
        #Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False
            
        #Unfreeze layer 4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True
            
        #Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
        
    def forward(self,x):
        x = self.model(x)
        return x



def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
    ])
    #Since we have trained the mode in batches we need to unsqueeze it inorder to make the dimensions as 
    #from [3,224,224] to [1,3,224,224]
    image_tensor = transform(image).unsqueeze(0)
    global trained_model
    if trained_model is None:
        trained_model = CarClassifierResNet()
        #Model is trained and saved using GPU but since we are having CPU we need to port it 
        trained_model.load_state_dict(torch.load("../model.pth",map_location=torch.device('cpu')))
        trained_model.eval()
        
    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted = torch.max(output,1)
        return class_names[predicted.item()]