from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Move the model to GPU
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Load the Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Use NVIDIA GPU

# Generate an image from a text prompt
prompt = "A futuristic cityscape at sunset"
with torch.autocast("cuda"):
    image = pipe(prompt).images[0]

# Save the generated image
image.save("generated_image.png")
print("Image saved as generated_image.png")
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Function to generate and display the image
def generate_image():
    prompt = entry.get()
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]
    image.save("generated_image.png")
    img = Image.open("generated_image.png")
    img = img.resize((512, 512), Image.Resampling.LANCZOS)  # Updated line
    img_tk = ImageTk.PhotoImage(img)
    label.config(image=img_tk)
    label.image = img_tk

# Create the GUI
root = tk.Tk()
root.title("Text-to-Image Generator")
root.geometry("600x700")

# Input field for the prompt
entry = tk.Entry(root, width=50)
entry.pack(pady=10)

# Button to generate the image
generate_button = tk.Button(root, text="Generate Image", command=generate_image)
generate_button.pack(pady=10)

# Label to display the generated image
label = tk.Label(root)
label.pack(pady=10)

# Run the application
root.mainloop()