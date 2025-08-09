import gradio as gr
from openai import OpenAI
import re

# -------------------
# 모델/토큰 예산 설정
# -------------------
MODEL = "gpt-5-chat-latest"

# 출력 토큰 상한 (네 설정)
MAX_OUTPUT_TOKENS = 512

# 입력 히스토리 토큰 예산
# 모델 전체 컨텍스트가 크더라도 안전하게 여유를 두고 잘라내자
MAX_INPUT_TOKENS = 6000           # 필요하면 2~8k 사이로 조절
SAFETY_MARGIN_TOKENS = 256        # 시스템/메타/오차 여유

# OpenAI API 키 & 클라이언트
with open("token.txt", "r", encoding="utf-8") as f:
    API_KEY = f.read().strip()
client = OpenAI(api_key=API_KEY)

# ---- 프롬프트 로더 ----
def load_prompts():
    with open("assets/prompt/system-message.txt", "r", encoding="utf-8") as f:
        sys_msg = f.read().strip()
    with open("assets/prompt/prompt-message.txt", "r", encoding="utf-8") as f:
        pmt_msg = f.read().strip()
    return sys_msg, pmt_msg

# ---- 화면에 뿌릴 때 system 숨기기 ----
def ui_view(history):
    return [m for m in history if m["role"] != "system"]

# -------------------------
# 토큰 카운터 (의존성 없이 대략치)
# -------------------------
_ENCODER = None  # tiktoken 제거

def count_message_tokens(messages, model=MODEL):
    """
    대략적인 메시지 토큰 수 추정 (의존성 없는 간단 버전).
    4문자 ≈ 1토큰 가정 + 메시지당 오버헤드(+4) + priming(+2)
    """
    total = 0
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        total += 4  # message-level overhead
        total += max(1, len(role) // 4)
        total += max(1, len(content) // 4)
    total += 2
    return total

# --------------------------------------
# 히스토리 슬라이싱: 오래된 것부터 잘라서 예산 맞추기
# - system + 최초 assistant(상황설명)는 항상 보존
# - 최신 대화는 최대한 보존
# --------------------------------------
def trim_history_for_budget(history, model=MODEL,
                            max_input_tokens=MAX_INPUT_TOKENS,
                            safety=SAFETY_MARGIN_TOKENS):
    if not history:
        return history

    # 보존해야 할 sticky 인덱스: 모든 system + 최초 assistant(상황 설명)
    sticky_idx = {i for i, m in enumerate(history) if m.get("role") == "system"}
    first_asst = next((i for i, m in enumerate(history) if m.get("role") == "assistant"), None)
    if first_asst is not None:
        sticky_idx.add(first_asst)

    # 입력 예산 = 설정값 - 출력 토큰 - 세이프티
    budget = max(512, max_input_tokens - MAX_OUTPUT_TOKENS - safety)

    # 이미 예산 이하면 그대로 반환
    if count_message_tokens(history, model) <= budget:
        return history

    # 오래된 것부터 제거 (sticky는 건너뜀)
    trimmed = history[:]
    i = 0
    while count_message_tokens(trimmed, model) > budget and i < len(trimmed):
        # 최신 대화 2개(일반적으로 user/assistant)는 가능하면 남겨두자
        if len(trimmed) - i <= 2:
            break
        if i in sticky_idx:
            i += 1
            continue
        del trimmed[i]
    return trimmed

def trim_incomplete_sentence(text: str) -> str:
    # 문장 종결부호를 기준으로 가장 마지막 완성 문장만 남김
    # 한국어 종결 + 영어 구두점 모두 감지
    match = re.search(r'(.+[.?!…]|.+다\.|.+요\.|.+네\.)', text, re.DOTALL)
    if match:
        return match.group(1)
    return text  # 종결부호 없으면 원문 그대로

# ---- 초기화 (state용: system 포함, 화면용: system 제외) ----
def init_chat():
    sys_msg, pmt_msg = load_prompts()
    history = [
        {"role": "system", "content": sys_msg},
        {"role": "assistant", "content": pmt_msg},  # 시작할 때 보여줄 '상대편 대답'
    ]
    return ui_view(history), history

# ---- 리셋(쓰레기통 버튼) ----
def reset_chat():
    return init_chat()

# ---- 전송/스트리밍 (generator, openai>=1.x) ----
def on_submit(user_message, history):
    if not user_message:
        yield ui_view(history), history
        return

    # 1) 유저 입력을 즉시 UI에 반영
    local = history + [{"role": "user", "content": user_message}]
    # API 호출 전에 입력 예산 맞춰 슬라이싱
    local = trim_history_for_budget(local, model=MODEL)

    ui = ui_view(local + [{"role": "assistant", "content": ""}])  # 빈 자리
    yield ui, local  # ← 사용자 메시지가 바로 보임

    # 2) OpenAI 스트리밍 (신규 SDK)
    partial = ""
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=local,
            temperature=0.7,
            max_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        )
        for chunk in stream:
            delta = getattr(chunk.choices[0].delta, "content", None)
            if delta:
                partial += delta
                ui[-1]["content"] = partial
                yield ui, local
    except Exception as e:
        partial = f"(오류) {e}"
        ui[-1]["content"] = partial
        yield ui, local

    # 3) 최종 history 확정
    partial = trim_incomplete_sentence(partial)  # 미완성 문장 제거
    local = local + [{"role": "assistant", "content": partial}]
    yield ui_view(local), local


# =========================
# Gradio 인터페이스
# =========================
with gr.Blocks() as demo:
    gr.Markdown("## ChatGPT Chatbot")
    chatbot = gr.Chatbot(type="messages", height=720)
    msg = gr.Textbox(placeholder="메시지를 입력해 대화해보세요!")
    state = gr.State([])  # system 포함 풀 history

    # 앱 로드시 초기 assistant 메시지 뿌리기
    demo.load(fn=init_chat, inputs=None, outputs=[chatbot, state])

    # 전송 핸들러(제너레이터 연결)
    msg.submit(on_submit, inputs=[msg, state], outputs=[chatbot, state])
    msg.submit(lambda: "", None, msg)  # 입력창 비우기

    # 쓰레기통(초기화)
    chatbot.clear(fn=reset_chat, outputs=[chatbot, state])

demo.launch(share=True)


