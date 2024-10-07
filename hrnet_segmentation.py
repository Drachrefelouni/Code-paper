
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from hrnet_segmentation import get_hrnet_model  # Load your HRNet model

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained HRNet segmentation model and move it to GPU
model = get_hrnet_model()  # Load your model here
model = model.to(device)
model.eval()

# Load and preprocess the image
image = Image.open('image.jpg')
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image).unsqueeze(0).to(device)  # Move tensor to GPU

# Perform inference on GPU
with torch.no_grad():
    output = model(input_tensor)['out'][0]
output_predictions = output.argmax(0).byte().cpu().numpy()  # Move result back to CPU

# Save segmentation mask
Image.fromarray(output_predictions).save('segmented_image.png')
