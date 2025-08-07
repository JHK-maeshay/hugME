import gradio as gr
import openai

# OpenAI API 키 로드
with open("token.txt", "r") as f:
    openai.api_key = f.read().strip()

# 메시지 이력 저장용 상태 변수
def chat(user_message, history):
    if history is None:
        history = []

    messages = [{"role": "system",
                 "content": "You are a helpful assistant."}]
    for user, bot in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})
    messages.append({"role": "user", "content": user_message})

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )

    reply = response.choices[0].message.content.strip()
    history.append((user_message, reply))
    return history, history

# 상태 초기화 함수
def reset_chat():
    return [], []

# Gradio 인터페이스 정의
with gr.Blocks() as demo:
    gr.Markdown("## ChatGPT Chatbot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="메시지를 입력해 대화해보세요!")
    state = gr.State([])

    def on_submit(message, history):
        return chat(message, history)

    msg.submit(on_submit, inputs=[msg, state], outputs=[chatbot, state])
    msg.submit(lambda: "", None, msg)  # 입력창 초기화

    chatbot.clear(fn=reset_chat, outputs=[chatbot, state])

demo.launch()
