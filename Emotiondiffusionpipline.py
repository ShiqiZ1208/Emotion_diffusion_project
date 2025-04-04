import torch
from diffusers import StableDiffusionPipeline
from Emotion_detection import emodetect, diasum, BARTsum
from color_transfer import lab_color_transfer
from Model.Lora import Lora_fine_tuning_BART
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def sdpipline(content, sumckpt, emockpt, mode = "plain"):
  #with open(file_path, 'r') as file:
    #content = file.read()

  if mode.lower() == "plain":
      summary = BARTsum(content)
  elif mode.lower() == "dialogue":
      summary = diasum(sumckpt, content)

  emotion = emodetect(emockpt, summary)
  print(f"summary:\n{summary}\nEmotion: {emotion}")

  prompt = f"A emotion based painting that vividly captures the essence of '{summary}' with a mood of '{emotion}'."

  backcolor = "A background color express emotion of {emotion}"

  sd_model_id = "runwayml/stable-diffusion-v1-5"
  pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
  pipe = pipe.to(device)

  image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
  color = pipe(backcolor, num_inference_steps=50, guidance_scale=7.5).images[0]

  np_content_image = np.array(image)
  np_color_image = np.array(color)

  alpha = 0.5
  final_image = lab_color_transfer(np_content_image, np_color_image, alpha)

  plt.figure(figsize=(8, 8))
  plt.imshow(final_image)
  plt.axis("off")
  plt.title("Final Generated Image")
  plt.show()

  plt.figure(figsize=(8, 8))
  plt.imshow(np_content_image)
  plt.axis("off")
  plt.title("content image without color transfer")
  plt.show()

  plt.figure(figsize=(8, 8))
  plt.imshow(np_color_image)
  plt.axis("off")
  plt.title("backgorund color")
  plt.show()

  return final_image, np_content_image, np_color_image

