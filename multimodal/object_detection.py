# Import necessary libraries for object detection
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import OwlViTForObjectDetection, OwlViTProcessor

# Load the OwlViT processor and model from Hugging Face
processor = OwlViTProcessor.from_pretrained(
    "google/owlvit-base-patch32", use_fast=False
)
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Define the URL of the image to analyze
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# Open the image from the URL
image = Image.open(requests.get(url, stream=True).raw)

# Define the text queries for object detection
texts = [["a photo of a cat", "a photo of a dog", "remote control", "cat tail"]]

# Process the image and text inputs for the model
inputs = processor(text=texts, images=image, return_tensors="pt")
# Get the model outputs
outputs = model(**inputs)

# Prepare target sizes for post-processing
target_sizes = torch.Tensor([image.size[::-1]])
# Post-process the model outputs to get detection results
results = processor.post_process_object_detection(
    outputs=outputs, target_sizes=target_sizes, threshold=0.1
)

# Retrieve predictions for the first image for the corresponding text queries
i = 0
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# Create a draw object to annotate the image
draw = ImageDraw.Draw(image)

# Load a default font for drawing text
font = ImageFont.load_default()

# Draw each bounding box and label on the image
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]  # Round box coordinates
    label_text = text[label]  # Get the label text
    print(
        f"Detected {label_text} with confidence {round(score.item(), 3)} at location {box}"
    )
    # Draw the bounding box on the image
    draw.rectangle(box, outline="red")
    # Draw the label above the bounding box
    draw.text((box[0], box[1]), label_text, fill="red", font=font)

# Save the annotated image as output.png
image.save("output.png")
