import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import os
# Load the saved model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust to match the original model's output units
model.load_state_dict(torch.load('potato_leaf_diseases.pth'))
model.eval()

# Create a new model with the correct final layer
new_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
new_model.fc = nn.Linear(new_model.fc.in_features, 4)  # Adjust to match the desired output units

# Copy the weights and biases from the loaded model to the new model
new_model.fc.weight.data = model.fc.weight.data[0:2]  # Copy only the first 2 output units
new_model.fc.bias.data = model.fc.bias.data[0:2]

#get the test directory
test_directory = "test_dataset"

#iterate over files in that directory
for filename in os.listdir(test_directory):
    f=os.path.join(test_directory,filename)
    if os.path.isfile(f):

# Load and preprocess the unseen image
        image_path = f  # Replace with the path to your image
        image = Image.open(image_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

# Perform inference
        with torch.no_grad():
            output = model(input_batch)

# Get the predicted class

            _, predicted_class = output.max(1)

# Map the predicted class to the class name
        class_names = ['Potato Early blight', 'Potato Healthy', 'Potato Late blight']  # Make sure these class names match your training data
       
        predicted_class_name = class_names[predicted_class.item()]

        print(f'File: {f}, The predicted class is: {predicted_class_name}')


