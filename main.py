#!pip install transformers diffusers torch

from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

# Load model from Hugging Face
model_name = "liamhvn/disney-pixar-cartoon-b"
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
# pipe = pipe.to("cuda") # using gpu nvidia
# pipe.enable_attention_slicing()
# pipe.enable_sequential_cpu_offload()

# PROMPT VARIABLES
prompt = "Generate this person in a Pixar cartoon style; keep hair color in brown spectrum; male gender"
image_path="./image.jpeg"
input_image = Image.open(image_path)

#ultimate combination
# Define different numbers of inference steps
# guidance_scales = [7.5, 10.0, 12.5, 15.0]
guidance_scales = [7.5]
steps_list = [75, 100]
outputs = []
strengths = [0.25, 0.5]  # Different levels of style intensity
seeds = [64, 128, 256, 512]
name = []

# for seed in seeds:
# generator=torch.manual_seed(seed)
for scale in guidance_scales:
  for strength in strengths:
    for steps in steps_list:
        pixar_image = pipe(prompt, init_image=input_image, strength=strength, num_inference_steps=steps, guidance_scale=scale).images[0]
        outputs.append(pixar_image)
        name.append(f"var_scale{scale}+strg{strength}+steps{steps}")
        pixar_image.show()

# Save the outputs
for i, pixar_image in enumerate(outputs):
    pixar_image.save(f"pixar_output{name[i]}.png")
