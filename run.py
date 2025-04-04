import argparse
import os
from Emotiondiffusionpipline import sdpipline
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("-D", "--Document",)
parser.add_argument("-Epath", "--emockpt",)
parser.add_argument("-Spath", "--sumckpt",)
parser.add_argument("-mode", "--mode",)
parser.add_argument("-E", "--isEmotion",)
parser.add_argument("-S", "--isSummary",)
parser.add_argument("-C", "--iscolortransfer",)


args = parser.parse_args()
is_Emodetect = args.isEmotion
is_Summary = args.isSummary
is_color_transfer = args.iscolortransfer
file_path = args.Document
mode = args.mode
emockpt = args.emockpt
sumckpt = args.sumckpt


if is_Summary.lower() == "true":
  is_Summary = True
else:
  is_Summary = False

if is_color_transfer.lower() == "true":
  is_color_transfer = True
else:
  is_color_transfer = False

if is_Emodetect.lower() == "true":
  is_Emodetect = True
else:
  is_Emodetect = False

if mode.lower() == "dialogue":
  mode = "dialogue"
else:
  mode = "plain"

with open(file_path, 'r') as file:
    content = file.read()

print(type(is_Emodetect), is_Summary, is_color_transfer, mode, content)
final_image, np_content_image, np_color_image = sdpipline(content, sumckpt, emockpt, mode, is_Emodetect, is_Summary, is_color_transfer)

print(final_image.dtype)
print(np_content_image.dtype)

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


