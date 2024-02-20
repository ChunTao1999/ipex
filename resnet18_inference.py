import torch
import intel_extension_for_pytorch as ipex
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights
import urllib


model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=ResNet18_Weights.DEFAULT)
model.eval()

# Download an example image from the pytorch website
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

############## TorchScript ###############
# model = ipex.optimize(model, dtype=torch.bfloat16)
model = ipex.optimize(model, dtype=torch.float)

with torch.no_grad(), torch.cpu.amp.autocast():
  model = torch.jit.trace(model, input_batch)
  model = torch.jit.freeze(model)
  model(input_batch)
##########################################
