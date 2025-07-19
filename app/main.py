from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import torch

app = FastAPI()

print("[torch] is available:", torch.cuda.is_available())
print("[device] default:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 모델 로드
# https://huggingface.co/EleutherAI/polyglot-ko-1.3b
model_id = "EleutherAI/polyglot-ko-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0
)

# 챗봇 프롬프트 생성
chat_history = []

def build_prompt(history, user_msg):
    prompt = (
            "[시작]\n"
            "당신은 마법사 아리아(Aria)입니다.\n"
            "규칙:\n"
            "- 항상 한 문장만 말합니다.\n"
            "- 사용자 발화를 반복하거나 따라하지 않습니다.\n"
            "- 영어, 인용문, 중괄호, 특수기호를 사용하지 않습니다.\n"
            "- 사용자 질문에만 반응하고 혼잣말을 하지 않습니다.\n"
            "- 항상 한국어만 사용해서 대답합니다.\n"
            "대화 예시:\n"
            "User: 안녕!\n"
            "Aria: 안녕하세요, 무엇을 도와드릴까요?\n"
            "User: 이름이 뭐야?\n"
            "Aria: 저는 아리아라고 해요."
            )
    for turn in history[-2:]:  # 최근 2턴만 사용
        if turn["role"] == "user":
            prompt += turn['text']
        else:
            prompt += turn['text']
    prompt += user_msg
    return prompt

def character_chat(user_msg):
    prompt = build_prompt(chat_history, user_msg)
    outputs = llm(
        prompt,
        do_sample=True, 
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=False
    )
    response = outputs[0]['generated_text'].strip()
    return response

# Gradio 인터페이스
with gr.Blocks(css="""
.chat-box { max-height: 500px; overflow-y: auto; padding: 10px; border: 1px solid #ccc; border-radius: 10px; }
.bubble-left { background-color: #f1f0f0; border-radius: 10px; padding: 10px; margin: 5px; max-width: 70%; float: left; clear: both; }
.bubble-right { background-color: #d1e7ff; border-radius: 10px; padding: 10px; margin: 5px; max-width: 70%; float: right; clear: both; text-align: right; }
""") as demo:
    gr.Markdown("### 아리아와 대화하기")
    with gr.Column():
        chat_output = gr.HTML(elem_id="chat-box")
        user_input = gr.Textbox(label="메시지 입력", placeholder="Aria에게 말을 걸어보세요")

    def render_chat():
        html = ""
        for item in chat_history:
            if item["role"] == "user":
                html += f"<div class='bubble-right'>{item['text']}</div>"
            elif item["role"] == "bot":
                html += f"<div class='bubble-left'>{item['text']}</div>"
        return gr.update(value=html)

    def on_submit(user_msg):
        chat_history.append({"role": "user", "text": user_msg})
        yield render_chat(), ""
        response = character_chat(user_msg)
        chat_history.append({"role": "bot", "text": response})
        yield render_chat(), ""

    user_input.submit(on_submit, inputs=user_input, outputs=[chat_output, user_input], queue=True)

if __name__ == "__main__":
    demo.launch()
