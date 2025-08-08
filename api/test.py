import gradio as gr
import openai

# 사용 모델
MODEL = "gpt-5-chat-latest"

# OpenAI API 키 로드
with open("token.txt", "r") as f:
    openai.api_key = f.read().strip()

# ---- 프롬프트 로더 ----
def load_prompts():
    with open("assets/prompt/system-message.txt", "r", encoding="utf-8") as f:
        sys_msg = f.read().strip()
    with open("assets/prompt/prompt-message.txt", "r", encoding="utf-8") as f:
        pmt_msg = f.read().strip()
    return sys_msg, pmt_msg

# ---- history 초기화 (state용: system 포함, 화면용: system 제외) ----
def init_chat():
    sys_msg, pmt_msg = load_prompts()
    history = [
        {"role": "system", "content": sys_msg},
        {"role": "assistant", "content": pmt_msg},  # 시작할 때 보여줄 '상대편 대답'
    ]
    return ui_view(history), history  # (chatbot 출력용, state)

# ---- 화면에 뿌릴 때 system 숨기기 ----
def ui_view(history):
    return [m for m in history if m["role"] != "system"]

# ---- 대화 함수 ----
def chat(user_message, history):
    # history는 state(=system 포함)로 들어옴
    if not user_message:
        return ui_view(history), history

    # 1) 유저 입력 추가
    history = history + [{"role": "user", "content": user_message}]

    # 2) OpenAI 호출
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=history,
        temperature=0.7,
        max_tokens=96
    )
    assistant_reply = resp.choices[0].message.content.strip()

    # 3) 답변 추가
    history = history + [{"role": "assistant", "content": assistant_reply}]

    # 4) 화면/상태 반환
    return ui_view(history), history

# ---- 리셋(쓰레기통 버튼) ----
def reset_chat():
    chatbot_view, state = init_chat()
    return chatbot_view, state

# =========================
# Gradio 인터페이스
# =========================
with gr.Blocks() as demo:
    gr.Markdown("## ChatGPT Chatbot")
    # Chatbot은 messages 모드로 쓰는게 최신권장
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="메시지를 입력해 대화해보세요!")
    state = gr.State([])  # system 포함 풀 history를 여기에 유지

    # 앱 로드시 초기 assistant 메시지 뿌리기
    demo.load(fn=init_chat, inputs=None, outputs=[chatbot, state])

    # 전송 핸들러
    def on_submit(message, history):
        return chat(message, history)

    msg.submit(on_submit, inputs=[msg, state], outputs=[chatbot, state])
    msg.submit(lambda: "", None, msg)  # 입력창 비우기

    # 쓰레기통(초기화) -> system+assistant 초기상태로 재시작
    chatbot.clear(fn=reset_chat, outputs=[chatbot, state])

demo.launch()
