# Project Name: Emotion Detection & image generation Pipeline

## Description
This project provides an emotion detection and text summarization pipeline built with BART and other natural language processing (NLP) techniques. It can summarize long texts and detect emotions, making it suitable for applications such as sentiment analysis, content summarization, and chatbot responses.

## Table of Contents
1. [Installation](#installation)
2. [usage](#usage)
3. [Contributing](#contributing)
4. [License](#license)
5. [Contact](#contact)

## Installation
Follow these steps to set up the project locally.

1. Clone the repository:
   ```bash
   git clone https://github.com/ShiqiZ1208/Emotion_diffusion_project.git
   cd Emotion_diffusion_project
    ```
## usage

1. import function from Emotiondiffusionpipline.py
   ```bash
   from Emotiondiffusionpipline import sdpipline
    ```
2. load checkpoint from google drive [https://drive.google.com/drive/folders/11yC9lALzwwHdbJUIPmhgwB96H1rLqZTt?usp=drive_link]
3. run the code
   ```bash
   dialogue = """...
   ...
   """
   sumckpt = "lora_bart1GAN_0_5.pth"
   emockpt = "Saved_Model"
   final_image, np_content_image, np_color_image = sdpipline(dialogue, sumckpt, emockpt, mode = "dialogue")
   ```
