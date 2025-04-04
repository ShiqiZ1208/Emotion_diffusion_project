import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed, pipeline, BartForConditionalGeneration, BartTokenizer
from Model.Lora import BART_base_model, Lora_fine_tuning_BART, BERT_base_model, Lora_fine_tuning_BERT
import nltk

seed = 0
torch.manual_seed(seed)
set_seed(seed)

def emodetect(ckpt, text):
    emotion_model_path = ckpt

    emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_path)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_path)
    emotion_classifier = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer)

    nltk.download('punkt')
    nltk.download('punkt_tab')
    go_emotions_mapping = {
        0: 'admiration',
        1: 'amusement',
        2: 'anger',
        3: 'annoyance',
        4: 'approval',
        5: 'caring',
        6: 'confusion',
        7: 'curiosity',
        8: 'desire',
        9: 'disappointment',
        10: 'disapproval',
        11: 'disgust',
        12: 'embarrassment',
        13: 'excitement',
        14: 'fear',
        15: 'gratitude',
        16: 'grief',
        17: 'joy',
        18: 'love',
        19: 'nervousness',
        20: 'optimism',
        21: 'pride',
        22: 'realization',
        23: 'relief',
        24: 'remorse',
        25: 'sadness',
        26: 'surprise',
        27: 'neutral'
    }
    emotion_output = emotion_classifier(text)
    label_str = emotion_output[0]['label']

    if label_str.startswith("LABEL_"):
        label_str = label_str.replace("LABEL_", "")
    emotion_label = int(label_str)

    # Map the numeric label to its corresponding emotion name
    emotion = go_emotions_mapping.get(emotion_label, "unknown")

    return emotion

def diasum(ckpt, dialogue):
    tokenizer = tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = torch.load(ckpt, weights_only=False)
    model.eval()
    input_ids = tokenizer(dialogue, truncation=True, padding='max_length', max_length=512, return_tensors= "pt")
    input_ids.to("cuda")
    model.to("cuda")
    output_ids = model.generate(**input_ids, max_length = 256)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return summary

def BARTsum(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary_output = summarizer(text, max_length=130, min_length=30, do_sample=False)
    summary = summary_output[0]['summary_text']

    return summary

