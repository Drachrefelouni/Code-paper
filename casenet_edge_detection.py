
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from casenet import get_casenet_model  # Load your CASENet model

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained CASENet model and move it to GPU
model = get_casenet_model()  # Load your model here
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
    output = model(input_tensor)
edge_map = output[0].cpu().numpy()  # Move result back to CPU

# Save edge detection result
edge_map_image = (edge_map[0] * 255).astype(np.uint8)
Image.fromarray(edge_map_image).save('edge_map.png')
