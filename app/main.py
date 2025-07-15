from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from transformers import pipeline
from PIL import Image
import re, os
import gradio as gr

app = FastAPI()

# 1. LLM 파이프라인 초기화 (SmolLM3 모델)
llm = pipeline("text-generation", model="HuggingFaceTB/SmolLM3-3B")

# 2. 감정 및 상황별 이미지 매핑
emotion_to_face = {
    "happy": "aria_happy.png",
    "sad": "aria_sad.png",
    "angry": "aria_angry.png",
    "excited": "aria_excited.png",
    "nervous": "aria_nervous.png",
    "neutral": "aria_neutral.png"
}
situation_to_bg = {
    "greeting": "bg_town.jpg",
    "mission_start": "bg_forest_day.jpg",
    "enemy_detected": "bg_dungeon_dark.jpg",
    "spooky_location": "bg_cave.png",
    "farewell": "bg_sunset.jpg"
}

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
def combine_images(bg_path, char_path):
    bg = Image.open(bg_path).convert("RGBA")
    char = Image.open(char_path).convert("RGBA")
    char = char.resize((300, 300))
    pos = ((bg.width - char.width) // 2, bg.height - char.height - 20)
    bg.paste(char, pos, char)
    return bg

# 5. 챗봇 처리 함수 (Gradio용)
def character_chat(prompt):
    system_prompt = (
        "You are Aria, a cheerful and expressive fantasy mage."
        " Respond in multiple steps if needed."
        " Format: \"text\" (emotion: tag, situation: tag)"
    )
    full_prompt = system_prompt + "\nUser: " + prompt + "\nAria:"

    raw_output = llm(full_prompt, max_new_tokens=300)[0]['generated_text']
    parsed = parse_output(raw_output)

    result_outputs = []
    for i, item in enumerate(parsed):
        face = emotion_to_face.get(item['emotion'], "aria_neutral.png")
        bg = situation_to_bg.get(item['situation'], "bg_default.jpg")
        composite = combine_images(os.path.join("assets/bg", bg), os.path.join("assets/face", face))
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
    gr.Markdown("# Aria 캐릭터 챗봇")
    with gr.Column():
        chat_output = gr.HTML(value="<div class='chat-box' id='chat-box'></div>")
        user_input = gr.Textbox(label="Say something to Aria", placeholder="Type here and press Enter")

    def render_chat():
        html = "<div class='chat-box'>"
        for item in chat_history:
            if item['role'] == 'user':
                html += f"<div class='bubble-right'>{item['text']}</div>"
            elif item['role'] == 'bot':
                html += f"<div class='bubble-left'>{item['text']}<br><img class='image-preview' src='{item['image']}'></div>"
        html += "</div>"
        return html

    def on_submit(user_msg):
        chat_history.append({"role": "user", "text": user_msg})
        bot_results = character_chat(user_msg)
        for text, image_path in bot_results:
            chat_history.append({"role": "bot", "text": text, "image": image_path})
        return render_chat(), ""

    user_input.submit(on_submit, inputs=user_input, outputs=[chat_output, user_input])

if __name__ == "__main__":
    demo.launch()