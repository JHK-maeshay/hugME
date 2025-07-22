from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import torch
import re
import os

print("[torch] is available:", torch.cuda.is_available())
print("[device] default:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 모델 로드
model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"

# 허깅 페이스 secret에 등록된 토큰 로드
access_token = os.environ.get("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    token=access_token
)
model.eval()
if torch.cuda.is_available():
    model.to("cuda")
llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16
)

# 챗봇 프롬프트 생성
def build_prompt(history, user_msg, user_name="User", bot_name="Tanjiro"):
    with open("assets/prompt/init.txt", "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    for turn in history[-16:]:
        role = user_name if turn["role"] == "user" else bot_name
        prompt += f"{role}: {turn['text']}\n"

    prompt += f"{user_name}: {user_msg}\n"
    prompt += f"{bot_name}:"
    return prompt

# 출력에서 응답 추출
def extract_response(full_text, prompt, bot_name="Tanjiro"):
    if full_text.startswith(prompt):
        reply = full_text[len(prompt):].strip()
    else:
        reply = full_text.split(f"{bot_name}:")[-1].strip()
    user_token = "\nUser:"
    if user_token in reply:
        reply = reply.split(user_token)[0].strip()
    return reply

# 출력 생성 함수
def character_chat(user_msg, history):
    print("[debug] generationg...")
    prompt = build_prompt(history, user_msg)
    outputs = llm(
        prompt,
        do_sample=True,
        max_new_tokens=96,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=True
    )
    full_text = outputs[0]['generated_text']
    response = extract_response(full_text, prompt)
    return response

# 중단된 응답 여부 검사
def is_truncated_response(text: str) -> bool:
    return re.search(r"[.?!…\u2026\u2639\u263A\u2764\uD83D\uDE0A\uD83D\uDE22]$", text.strip()) is None

# 답변 유효성 검사
def is_valid_response(text: str, bot_name="Tanjiro", user_name="User") -> bool:
    if user_name + ":" in text:
        return False
    if bot_name + ":" in text:
        return False
    return True

# 답변 형식 정리
def clean_response(text: str, bot_name="Tanjiro"):
    return re.sub(rf"{bot_name}:\\s*", "", text).strip()

# Gradio 인터페이스
with gr.Blocks(css="""
.chat-box { max-height: 500px; overflow-y: auto; padding: 10px; border: 1px solid #ccc; border-radius: 10px; }
.bubble-left { background-color: #f1f0f0; border-radius: 10px; padding: 10px; margin: 5px; max-width: 70%; float: left; clear: both; }
.bubble-right { background-color: #d1e7ff; border-radius: 10px; padding: 10px; margin: 5px; max-width: 70%; float: right; clear: both; text-align: right; }
.reset-btn-container { text-align: right; margin-bottom: 10px; }
""") as demo:
    gr.Markdown("### 탄지로와 대화하기")
    with gr.Column():
        with gr.Row():
            gr.Markdown("")
            reset_btn = gr.Button("🔁 대화 초기화", elem_classes="reset-btn-container", scale=1)
        chat_output = gr.HTML(elem_id="chat-box")
        user_input = gr.Textbox(label="메시지 입력", placeholder="탄지로에게 말을 걸어보세요")
        state = gr.State([])

    def render_chat(history):
        html = ""
        for item in history:
            if item["role"] == "user":
                html += f"<div class='bubble-right'>{item['text']}</div>"
            elif item["role"] == "bot":
                html += f"<div class='bubble-left'>{item['text']}</div>"
        return gr.update(value=html)

    def on_submit(user_msg, history):
        history.append({"role": "user", "text": user_msg})
        html = render_chat(history)
        yield html, "", history

        #응답 생성
        while True:
            response = character_chat(user_msg, history)
            if is_valid_response(response):
                break
        response = clean_response(response)
        history.append({"role": "bot", "text": response})

        #중간에 응답이 끊긴 경우 추가 생성
        if is_truncated_response(response):
            while True:
                continuation = character_chat(response, history)
                if is_valid_response(continuation):
                    break
            continuation = clean_response(continuation)
            history.append({"role": "bot", "text": continuation})

        html = render_chat(history)
        yield html, "", history

    def reset_chat():
        return gr.update(value=""), "", []

    user_input.submit(on_submit, inputs=[user_input, state], outputs=[chat_output, user_input, state], queue=True)
    reset_btn.click(reset_chat, inputs=None, outputs=[chat_output, user_input, state])
    
    #허깅페이스에서 실행
    demo.launch()
    
