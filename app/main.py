from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from transformers import pipeline
from PIL import Image
import re, os
import gradio as gr
import torch

app = FastAPI()

# 1. LLM 파이프라인 초기화 (SmolLM3 모델)
print("[torch] is available:", torch.cuda.is_available())
print("[device] default:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
llm = pipeline("text-generation", model="HuggingFaceTB/SmolLM3-3B", device=0 if torch.cuda.is_available() else -1)

# 2. 감정 및 상황별 이미지 매핑
'''
이미지 매핑 예시
-----------------------------
이미지를 다음과 같이 집어넣으면 (./asset/face/)
happy.png
sad.png
angry.png
=>
이런 딕셔너리 형식으로 반환됨
{
    "happy": "happy.png",
    "sad": "sad.png",
    "angry": "angry.png"
}
=>
모델 출력의 감정 부분에 대응되는 이미지 출력
'''
def load_faces(face_dir="assets/face"):
    if not os.path.exists(face_dir):
        os.makedirs(face_dir)
    emotion_to_face = {}
    for filename in os.listdir(face_dir):
        if filename.endswith(".png"):
            emotion = os.path.splitext(filename)[0]  # 'happy.png' → 'happy'
            emotion_to_face[emotion] = filename      # "happy": "happy.png"
    return emotion_to_face

def load_bgs(bg_dir="assets/bg"):
    if not os.path.exists(bg_dir):
        os.makedirs(bg_dir)
    situation_to_bg = {}
    for filename in os.listdir(bg_dir):
        if filename.endswith(".png"):
            emotion = os.path.splitext(filename)[0]  # 'happy.png' → 'happy'
            situation_to_bg[emotion] = filename      # "happy": "happy.png"
    return situation_to_bg

emotion_to_face = load_faces()
situation_to_bg = load_bgs()

# 3. 출력 라인 파싱 함수
def parse_output(text: str):
    pattern = r'"(.*?)"\s*\(emotion:\s*(\w+),\s*situation:\s*(\w+)\)'
    results = []
    for line in text.strip().split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            results.append({
                "text": match.group(1),
                "emotion": match.group(2),
                "situation": match.group(3)
            })
    return results

# 4. 이미지 합성 함수
def combine_images(bg_path, face_path):
    try:
        bg = Image.open(bg_path).convert("RGBA")
    except FileNotFoundError:
        print(f"[warning] 배경 이미지 없음: {bg_path}")
        return None
    try:
        face = Image.open(face_path).convert("RGBA")
    except FileNotFoundError:
        print(f"[warning] 캐릭터 이미지 없음: {face_path}")
        return None
    # 이미지 합성
    bg.paste(face, (0, 0), face)
    return bg

# 5. 챗봇 처리 함수 (Gradio용)
'''
지금까지 대화 내용을 모두 프롬프트로 넣어서 대화내용을 기억하도록 함
'''
def build_prompt(chat_history, user_msg):
    system_prompt = (
        "You are Aria, a cheerful and expressive fantasy mage."
        " Respond in multiple steps if needed."
        " Format: \"text\" (emotion: tag, situation: tag)"
    )

    dialogue = ""
    for item in chat_history:
        if item["role"] == "user":
            dialogue += f"User: {item['text']}\n"
        elif item["role"] == "bot":
            dialogue += f"Aria: {item['text']}\n"

    dialogue += f"User: {user_msg}\nAria:"
    return system_prompt + "\n" + dialogue

def character_chat(prompt):
    full_prompt = build_prompt(chat_history, prompt)

    #raw_output = llm(full_prompt, max_new_tokens=300)[0]['generated_text']
    raw_output = '"우오아" (emotion: tag, situation: tag)'
    parsed = parse_output(raw_output)

    result_outputs = []
    for i, item in enumerate(parsed):
        face = emotion_to_face.get(item['emotion'], "neutral.png")
        bg = situation_to_bg.get(item['situation'], "default.jpg")
        composite = combine_images(os.path.join("assets/bg", bg), os.path.join("assets/face", face))
        img_path = None #이미지가 없으면 출력 안함
        if composite:
            img_path = f"static/output_{i}.png"
            composite.save(img_path)
        result_outputs.append((item['text'], img_path))

    return result_outputs

# 6. Gradio UI with chat history
chat_history = []

with gr.Blocks(css="""
.chat-box { max-height: 500px; overflow-y: auto; padding: 10px; border: 1px solid #ccc; border-radius: 10px; }
.bubble-left { background-color: #f1f0f0; border-radius: 10px; padding: 10px; margin: 5px; max-width: 70%; float: left; clear: both; }
.bubble-right { background-color: #d1e7ff; border-radius: 10px; padding: 10px; margin: 5px; max-width: 70%; float: right; clear: both; text-align: right; }
.image-preview { margin: 5px 0; max-width: 100%; border-radius: 10px; }
""") as demo:
    gr.Markdown("챗봇")
    with gr.Column():
        chat_output = gr.HTML(value="<div class='chat-box' id='chat-box'></div>")
        user_input = gr.Textbox(label="Say something to Aria", placeholder="Type here and press Enter")

    def render_chat():
        html = ""
        for item in chat_history:
            if item['role'] == 'user':
                html += f"<div class='bubble-right'>{item['text']}</div>"
            elif item['role'] == 'bot':
                bubble = f"<div class='bubble-left'>{item['text']}"
                if 'image' in item and item['image']:
                    bubble += f"<br><img class='image-preview' src='{item['image']}'>"
                bubble += "</div>"
                html += bubble
        return html

    def on_submit(user_msg):
        chat_history.append({"role": "user", "text": user_msg})

        bot_results = character_chat(user_msg)

        for item in bot_results:
            try:
                text, image_path = item  # unpack 시도
            except (ValueError, TypeError):
                # unpack 안되면 기본값 처리 (이미지 없이)
                text = str(item)
                image_path = None

            chat_entry = {"role": "bot", "text": text}
            if image_path:
                chat_entry["image"] = image_path

            chat_history.append(chat_entry)

        new_chat_html = render_chat()
        return f"<div class='chat-box' id='chat-box'>{new_chat_html}</div>", ""

    user_input.submit(on_submit, inputs=user_input, outputs=[chat_output, user_input])

if __name__ == "__main__":
    demo.launch()