import cv2
import easyocr
from ultralytics import YOLO
import re
import torch
import gradio as gr
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


reader = easyocr.Reader(['en'])


YOLO_MODEL_PATH = "./bilalmodel.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)

HF_MODEL_NAME = "magahcicek/avastin-side-effects-model"
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def run_inference(model, image_path, conf_threshold=0.5):
    results = model(image_path, conf=conf_threshold)
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, conf))
    return boxes

def extract_text(image, boxes, reader):
    results = []
    for (x1, y1, x2, y2, conf) in boxes:
        roi = image[y1:y2, x1:x2]
        ocr_results = reader.readtext(roi)
        combined_text = " ".join([res[1] for res in ocr_results])
        if combined_text:
            results.append(combined_text)
    return results

def clean_side_effects(text, drug_name):
    text = text.strip(" ,.\n\t")
    text = re.sub(re.escape(drug_name), '', text, flags=re.IGNORECASE)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r",+", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"([a-zA-Z])\.([a-zA-Z])", r"\1. \2", text)
    items = re.split(r"[,\n\-]+", text)
    items = [item.strip() for item in items if item.strip()]
    unique_items = []
    for item in items:
        if item.lower() not in [x.lower() for x in unique_items]:
            unique_items.append(item)
    unique_items = [item if item.endswith('.') else item + '.' for item in unique_items]
    return unique_items

def generate_side_effects(drug_name):
    prompt = f"[DRUG_START]{drug_name}[DRUG_END][SIDE_EFFECTS]"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True, padding="max_length").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        pad_token_id=tokenizer.eos_token_id,
        num_beams=5,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_side_effects(full_text, drug_name)


def pipeline(image):
    temp_image_path = os.path.join("/tmp", "temp_input.jpg")
    image.save(temp_image_path)
    img_cv2 = cv2.imread(temp_image_path)


    boxes = run_inference(yolo_model, temp_image_path)


    detected_texts = extract_text(img_cv2, boxes, reader)

    if not detected_texts:
        return "İlaç adı algılanamadı.", "Görselde ilaç ismi tespit edilemedi."


    drug_name = detected_texts[0]


    side_effects = generate_side_effects(drug_name)

    return drug_name, "\n".join(f"- {item}" for item in side_effects)


iface = gr.Interface(
    fn=pipeline,
    inputs=gr.Image(type="pil", label="Upload Medicine image"),
    outputs=[
        gr.Label(label="🧪 Drug Name"),
        gr.Textbox(label="💊 Predict Side Effect", lines=8)
    ],
    title="Drug Side Effect System",
    description="Detected Drug image otomatically find possible drug  side effects "
)

if __name__ == "__main__":
    iface.launch()
