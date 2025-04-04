import torch
from diffusers import StableDiffusionPipeline
from Emotion_detection import emodetect, diasum, BARTsum
from color_transfer import lab_color_transfer
from Model.Lora import BART_base_model, Lora_fine_tuning_BART, BERT_base_model, Lora_fine_tuning_BERT, custom_bart_loss
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 0
torch.manual_seed(0)

def sdpipline(content, sumckpt, emockpt, mode = "plain", is_emodetect = True, is_summary = True, is_color_transfer = True):
  #with open(file_path, 'r') as file:
    #content = file.read()
  summary = content
  emotion = 'neutral'
  if is_summary == True:
      if mode.lower() == "plain":
          summary = BARTsum(content)
      elif mode.lower() == "dialogue":
          summary = diasum(sumckpt, content)
  else:
      summary = content

  if is_emodetect == True:
    emotion = emodetect(emockpt, summary)
  
  print(f"summary:\n{summary}\nEmotion: {emotion}")

  if is_summary == True and is_emodetect == True:
    prompt = f"A emotion based painting that vividly captures the essence of '{summary}' with a mood of '{emotion}'."
  elif is_summary == False and is_emodetect == True:
    prompt = f"A emotion based painting that vividly captures the essence of '{summary}' with a mood of '{emotion}'."
  else:
    prompt = f"A emotion based painting that vividly captures the essence of '{summary}'."

  if is_color_transfer == True:
    backcolor = f"A background color express emotion of {emotion}"

  sd_model_id = "runwayml/stable-diffusion-v1-5"
  pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
  pipe = pipe.to(device)
  
  generator = torch.manual_seed(seed)
  image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
  if is_color_transfer == True:
    color = pipe(backcolor, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]

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
  else:
    np_content_image = np.array(image)
    final_image = np_content_image
    np_color_image = None
    
    plt.figure(figsize=(8, 8))
    plt.imshow(np_content_image)
    plt.axis("off")
    plt.title("content image without color transfer")
    plt.show()

  return final_image, np_content_image, np_color_image

