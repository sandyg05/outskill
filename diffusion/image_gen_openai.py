import base64
import os

import openai
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_image(
    prompt, size="1024x1024", model="gpt-image-1", output="output_2.png"
):
    """Generate an image from text prompt"""
    try:
        print(f"Attempting to generate image with model: {model}")

        # Try gpt-image-1 first
        response = client.images.generate(
            model=model, prompt=prompt, size=size, quality="medium", n=1
        )

        print(f"Success with {model}!")
        image_bytes = base64.b64decode(response.data[0].b64_json)
        with open(output, "wb") as f:
            f.write(image_bytes)

    except Exception as e:
        print(f"Error with {model}: {e}")

    return None


def edit_image(prompt, image_list, size="1024x1024", output="edited_2.png"):
    """Edit an image using a mask"""

    result = client.images.edit(
        model="gpt-image-1",
        image=[
            open(image_list[0], "rb"),
            open(image_list[1], "rb"),
        ],
        prompt=prompt,
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    # Save the image to a file
    with open(output, "wb") as f:
        f.write(image_bytes)

    return None


def edit_image_stamp(prompt, image_list, size="1024x1024", output="stamp_2.png"):

    result = client.images.edit(
        model="gpt-image-1", image=[open(image_list[0], "rb")], prompt=prompt
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    # Save the image to a file
    with open(output, "wb") as f:
        f.write(image_bytes)

    return None


def create_variation(image_path, size="1024x1024"):
    """Create a variation of an existing image"""
    with open(image_path, "rb") as image_file:
        response = client.images.create_variation(
            model="dall-e-2", image=image_file, size=size, n=1
        )
    return response.data[0].url


def download_image(url, filename):
    """Download image from URL"""
    if url is None:
        print("Cannot download image: URL is None")
        return

    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Image saved as {filename}")
    except Exception as e:
        print(f"Error downloading image: {e}")


# Example usage
if __name__ == "__main__":
    # ---------- Image Generation ---------- #

    # 1. Generate image - try gpt-image-1 first

    # print("Generating image...")
    # generate_image("Santa Claus wearing a traditional Indian kurta and dhoti, standing in front of a beautifully decorated house with Diwali lights and lanterns. He is lighting fireworks with children, surrounded by glowing diyas, colorful rangoli on the ground, and festive Indian sweets on a table. The night sky is lit with fireworks, blending Christmas cheer with Diwali joy.", model="gpt-image-1")

    # ---------- Image Editing ---------- #

    # 2. Edit image (you need to provide image.png and mask.png)
    # Note: gpt-image-1 doesn't support editing yet, so we use dall-e-2

    # print("Editing image 1...")

    # image_list = ['../assets/mokobara.jpeg','../assets/rayban.jpeg']

    # prompt = """
    # Generate a photorealistic image of a traveler in a beach resort
    # labeled 'Travel Light' with a ribbon and handwriting-like font,
    # containing all the items in the reference pictures.
    # """

    # edit_image(prompt, image_list)

    # print("Editing image 2...")
    # image_list = ['../assets/taj.jpg']
    # prompt = """
    # Create a bold, geometric travel stamp in the style of iconic stamps like Houston, Lyon, Nuuk, and Zaanse Schans. Use clean lines, balanced detail, and flat vector color design.
    # Style Requirements:

    # Simplify real-world objects from the photo into clear, geometric shapes with character
    # Use flat, vivid colors with a unique 3-5 color palette for visual harmony
    # Apply clean edges with no strokes unless decorative - rely on shape boundaries for clarity
    # Use flat fills only - no gradients, shadows, or lighting effects
    # Include 2-3 recognizable visual elements from the original image (objects, background elements, landmarks, or key structures) to convey the scene without clutter
    # Maintain a balanced and iconic composition - avoid overly abstract or empty results

    # Stamp Frame:

    # Design with perforated postage stamp edges
    # Center the stamp artwork to occupy 60-70 percent of the square canvas with consistent margins
    # Use a solid background color that's distinct from the stamp interior
    # No text or typography inside the stamp - rely purely on illustration

    # """
    # edit_image_stamp(prompt, image_list)

    # ---------- Image Variation ---------- #

    # 3. Create variation (you need to provide image.png)
    # Note: gpt-image-1 doesn't support variations yet, so we use dall-e-2
    print("Creating variation...")
    variation_url = create_variation("stamp.png")
    download_image(variation_url, "variation_2.png")
