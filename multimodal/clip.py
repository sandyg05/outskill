# Import necessary libraries
import requests  # requests is used to make HTTP requests
from PIL import Image  # PIL is used for image processing

# Import the CLIP model and processor from the transformers library
from transformers import CLIPModel, CLIPProcessor

# Load the pre-trained CLIP model and processor
# The model is used for image and text processing
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# The processor prepares the inputs for the model
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32", use_fast=False
)

# Define the URL of the image we want to analyze
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
url = "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg"

# Open the image from the URL using requests and PIL
# stream=True allows us to download the image without saving it to disk
image = Image.open(requests.get(url, stream=True).raw)

# Define the texts we want to compare with the image
texts = ["a photo of a cat", "a photo of a dog"]

# Process the inputs using the processor
# This prepares the text and image for the model
inputs = processor(
    text=texts,  # The texts we want to analyze
    images=image,  # The image we want to analyze
    return_tensors="pt",  # Return PyTorch tensors
    padding=True,  # Pad the inputs to the same length
)

# Pass the processed inputs to the model to get the outputs
outputs = model(**inputs)

# Extract the logits for the image from the model's outputs
logits_per_image = outputs.logits_per_image

# Apply softmax to the logits to get probabilities
# This converts the raw logits into probabilities that sum to 1
probs = logits_per_image.softmax(dim=1)

# Print the probabilities for each text in a formatted way
# This shows how likely the image is to match each text description
print(f"\n “a photo of a cat”: {probs[0][0] * 100:.2f}%")
print(f"\n “a photo of a dog”: {probs[0][1] * 100:.2f}%")
