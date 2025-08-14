# Import necessary libraries
import requests  # requests is used to make HTTP requests to fetch images
import torch  # PyTorch library for tensor operations and model handling
from PIL import Image  # PIL is used for opening and manipulating images
from transformers import (  # Importing the processor and model from Hugging Face Transformers
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

# Determine the device to run the model on: use GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained BLIP2 processor and model from Salesforce
# The processor prepares the inputs for the model,
# while the model generates answers based on the inputs
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,  # Load the model with half-precision for efficiency
)
model.to(device)  # Move the model to the specified device (GPU or CPU)

# Define the URL of the image we want to analyze
url = "http://images.cocodataset.org/val2017/000000039769.jpg"

# Open the image from the URL using requests and PIL
# stream=True allows us to download the image without saving it to disk
image = Image.open(requests.get(url, stream=True).raw)

# Define the prompt for the model, asking a specific question about the image
prompt = "Question: How many remotes are there? Answer:"

# Process the image and prompt text using the processor
# This prepares the inputs for the model, converting them into tensors
inputs = processor(images=image, text=prompt, return_tensors="pt").to(
    device, torch.float16  # Move the inputs to the same device as the model
)

# Generate the output from the model based on the processed inputs
outputs = model.generate(**inputs)

# Decode the generated output tokens into human-readable text
# skip_special_tokens=True ensures that we do not include any special tokens in the output
text = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Print the decoded answer, stripping any leading or trailing whitespace
print(text[0].strip())
