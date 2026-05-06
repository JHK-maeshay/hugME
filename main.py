import os
import json
import re
import datetime
import gradio as gr
from openai import OpenAI
import google.generativeai as genai

# =========================
# 경로/기본값
# =========================
CONFIG_DIR = "config"
EXPORT_DIR = "exports"
KEYS_PATH = os.path.join(CONFIG_DIR, "keys.json")
SETTINGS_PATH = os.path.join(CONFIG_DIR, "settings.json")

OPENAI_DEFAULT_MODEL = "gpt-5-chat-latest"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
OLLAMA_DEFAULT_MODEL = "my-gemma" #default: "gemma4:e2b"
OLLAMA_BASE_URL = "http://192.168.0.32:11434/v1"


# 로컬 토큰 파일 폴백 (OpenAI용)
TOKEN_TXT_FALLBACK = "token.txt"

# -------------------
# 기본 설정(초기값)
# -------------------
DEFAULT_SETTINGS = {
    "max_input_tokens": 512,
    "max_output_tokens": 8192,
    "safety_margin_tokens": 256
}

# =========================
# 유틸: 파일/디렉터리
# =========================
def ensure_dirs():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)

def load_json_safe(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json_safe(path, data):
    ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================
# 프롬프트 로더
# =========================
def load_prompts():
    with open("assets/prompt/system-message.txt", "r", encoding="utf-8") as f:
        sys_msg = f.read().strip()
    with open("assets/prompt/prompt-message.txt", "r", encoding="utf-8") as f:
        pmt_msg = f.read().strip()
    return sys_msg, pmt_msg

# ---- 화면에 뿌릴 때 system 숨기기 ----
def ui_view(history):
    return [m for m in history if m.get("role") != "system"]

# -------------------------
# 토큰 카운터 (대략치)
# -------------------------
def count_message_tokens(messages):
    total = 0
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        total += 4
        total += max(1, len(role)//4)
        total += max(1, len(content)//4)
    total += 2
    return total

# --------------------------------------
# 히스토리 슬라이싱
# --------------------------------------
def trim_history_for_budget(history, max_input_tokens, max_output_tokens, safety):
    if not history:
        return history

    sticky_idx = {i for i, m in enumerate(history) if m.get("role") == "system"}
    first_asst = next((i for i, m in enumerate(history) if m.get("role") == "assistant"), None)
    if first_asst is not None:
        sticky_idx.add(first_asst)

    budget = max(512, max_input_tokens - max_output_tokens - safety)
    if count_message_tokens(history) <= budget:
        return history

    trimmed = history[:]
    i = 0
    while count_message_tokens(trimmed) > budget and i < len(trimmed):
        if len(trimmed) - i <= 2:
            break
        if i in sticky_idx:
            i += 1
            continue
        del trimmed[i]
    return trimmed

def trim_incomplete_sentence(text: str) -> str:
    match = re.search(r'(.+[.?!…]|.+다\.|.+요\.|.+네\.)', text, re.DOTALL)
    if match:
        return match.group(1)
    return text

# ---------- 포맷 빌더 ----------
def build_openai_messages(history):
    return history

def build_gemini_contents(history):
    sys_msgs = [m["content"] for m in history if m.get("role") == "system"]
    system_instruction = "\n\n".join(sys_msgs) if sys_msgs else ""
    contents = []
    for m in history:
        role = m.get("role")
        if role == "system":
            continue
        g_role = "user" if role == "user" else ("model" if role == "assistant" else "user")
        contents.append({"role": g_role, "parts": [m.get("content", "") or ""]})
    return system_instruction, contents

# =========================
# 상태 초기화/리셋
# =========================
def init_chat():
    ensure_dirs()
    sys_msg, pmt_msg = load_prompts()
    history = [
        {"role": "system", "content": sys_msg},
        {"role": "assistant", "content": pmt_msg},
    ]
    # 설정 로드(없으면 기본)
    settings = load_json_safe(SETTINGS_PATH, DEFAULT_SETTINGS.copy())
    return ui_view(history), history, settings["max_input_tokens"], settings["max_output_tokens"], settings["safety_margin_tokens"]

def reset_chat():
    ui, history, max_in, max_out, safety = init_chat()
    return ui, history

# =========================
# 스트리밍 어댑터
# =========================
def stream_openai(local_history, model, openai_key, max_output_tokens):
    client = OpenAI(api_key=openai_key)
    stream = client.chat.completions.create(
        model=model,
        messages=build_openai_messages(local_history),
        temperature=0.7,
        max_tokens=int(max_output_tokens),
        stream=True,
    )
    for chunk in stream:
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            yield delta

def stream_gemini(local_history, model_name, gemini_key, max_output_tokens):
    genai.configure(api_key=gemini_key)
    system_instruction, contents = build_gemini_contents(local_history)
    model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
    response = model.generate_content(
        contents,
        generation_config={"temperature": 0.7, "max_output_tokens": int(max_output_tokens)},
        stream=True,
    )
    for chunk in response:
        if hasattr(chunk, "text") and chunk.text:
            yield chunk.text

def stream_ollama(local_history, model, url, max_output_tokens):
    # Ollama는 OpenAI 호환 API를 제공하므로 base_url을 로컬 호스트로 변경하여 사용합니다.
    client = OpenAI(base_url=url, api_key="ollama-local") 
    stream = client.chat.completions.create(
        model=model,
        messages=build_openai_messages(local_history),
        temperature=0.7,
        max_tokens=int(max_output_tokens),
        stream=True,
    )
    for chunk in stream:
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            yield delta

# =========================
# 키/설정 로드 헬퍼
# =========================
def load_keys():
    data = load_json_safe(KEYS_PATH, {})
    # 구조 예시: {"openai": {"api_key": "...", "model": "..."}, "gemini": {"api_key": "...", "model": "..."}}
    return data

def resolve_openai_key():
    data = load_keys() or {}
    k = (data.get("openai") or {}).get("api_key") or ""
    if not k and os.path.exists(TOKEN_TXT_FALLBACK):
        with open(TOKEN_TXT_FALLBACK, "r", encoding="utf-8") as f:
            k = f.read().strip()
    return k

def resolve_gemini_key():
    data = load_keys() or {}
    return (data.get("gemini") or {}).get("api_key") or ""

def resolve_models():
    data = load_keys() or {}
    openai_model = (data.get("openai") or {}).get("model") or OPENAI_DEFAULT_MODEL
    gemini_model = (data.get("gemini") or {}).get("model") or GEMINI_DEFAULT_MODEL
    ollama_url = (data.get("ollama") or {}).get("base_url") or OLLAMA_BASE_URL
    ollama_model = (data.get("ollama") or {}).get("model") or OLLAMA_DEFAULT_MODEL
    return openai_model, gemini_model, ollama_url, ollama_model

def load_settings():
    s = load_json_safe(SETTINGS_PATH, DEFAULT_SETTINGS.copy())
    # 필수 키 보정
    for k, v in DEFAULT_SETTINGS.items():
        s.setdefault(k, v)
    return s

# =========================
# 대화 내보내기
# =========================
def export_chat(history, include_system=False, mode="flat"):
    ensure_dirs()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(EXPORT_DIR, f"chat_transcript_{ts}.txt")

    lines = []
    for m in history:
        role = (m.get("role") or "").strip()
        if role == "system" and not include_system:
            continue
        if role not in ("system", "user", "assistant"):
            role = "user"

        content = (m.get("content") or "").rstrip()

        if mode == "block":
            lines.append(f"{role}:")
            if content:
                lines.append(content)
        else:
            flat = re.sub(r"\s+", " ", content).strip()
            lines.append(f"{role}: {flat}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path

# =========================
# 전송/스트리밍 (탭1)
# =========================
# =========================
# 전송/스트리밍 (탭1)
# =========================
def on_submit(user_message, history, provider, max_input_tokens, max_output_tokens, safety_margin):
    if not user_message:
        return ui_view(history), history

    # 모델/키 결정은 JSON에서 로드
    openai_model, gemini_model, ollama_url, ollama_model = resolve_models()
    openai_key = resolve_openai_key()
    gemini_key = resolve_gemini_key()

    if provider == "OpenAI ChatGPT" and not openai_key:
        err = "(오류) OpenAI API 키가 설정되어 있지 않습니다."
        local = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": err}]
        return ui_view(local), local
    if provider == "Google Gemini" and not gemini_key:
        err = "(오류) Gemini API 키가 설정되어 있지 않습니다."
        local = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": err}]
        return ui_view(local), local

    # 사용자 입력 반영 + 예산 슬라이스
    local = history + [{"role": "user", "content": user_message}]
    local = trim_history_for_budget(local, int(max_input_tokens), int(max_output_tokens), int(safety_margin))

    # UI에 자리 만들기
    ui = ui_view(local + [{"role": "assistant", "content": "*응답 대기 중입니다. 잠시만 기다려 주십시오.*"}])
    
    # API 호출 대기 중 사용자 메시지 즉시 렌더링
    yield ui, local
    
    partial = ""
    try:
        if provider == "OpenAI ChatGPT":
            for delta in stream_openai(local, openai_model, openai_key, int(max_output_tokens)):
                partial += delta
                ui[-1]["content"] = partial
                yield ui, local
        elif provider == "Google Gemini":
            for delta in stream_gemini(local, gemini_model, gemini_key, int(max_output_tokens)):
                partial += delta
                ui[-1]["content"] = partial
                yield ui, local
        elif provider == "Local Ollama (Gemma 4.0)":
            for delta in stream_ollama(local, ollama_model, ollama_url, int(max_output_tokens)):
                partial += delta
                ui[-1]["content"] = partial
                yield ui, local
    except Exception as e:
        partial = f"(오류) {e}"
        ui[-1]["content"] = partial
        yield ui, local

    partial = trim_incomplete_sentence(partial)
    local = local + [{"role": "assistant", "content": partial}]
    yield ui_view(local), local

# =========================
# 탭2 토큰 저장/불러오기 핸들러
# =========================
def save_keys(openai_key, openai_model, gemini_key, gemini_model, ollama_url, ollama_model):
    data = {
        "openai": {"api_key": (openai_key or "").strip(), "model": (openai_model or OPENAI_DEFAULT_MODEL).strip()},
        "gemini": {"api_key": (gemini_key or "").strip(), "model": (gemini_model or GEMINI_DEFAULT_MODEL).strip()},
        "ollama": {"base_url": (ollama_url or OLLAMA_BASE_URL).strip(), "model": (ollama_model or OLLAMA_DEFAULT_MODEL).strip()}
    }
    save_json_safe(KEYS_PATH, data)
    return "✅ 저장됨: config/keys.json"

def load_keys_ui():
    data = load_keys() or {}
    openai_model = (data.get("openai") or {}).get("model") or OPENAI_DEFAULT_MODEL
    gemini_model = (data.get("gemini") or {}).get("model") or GEMINI_DEFAULT_MODEL
    ollama_url = (data.get("ollama") or {}).get("base_url") or OLLAMA_BASE_URL
    ollama_model = (data.get("ollama") or {}).get("model") or OLLAMA_DEFAULT_MODEL
    return openai_model, gemini_model, ollama_url, ollama_model, "🔄 불러오기 완료(키는 보안상 표시하지 않습니다)"

# =========================
# 탭3 설정 저장/불러오기 핸들러
# =========================
def save_settings(max_in, max_out, safety):
    s = {
        "max_input_tokens": int(max_in),
        "max_output_tokens": int(max_out),
        "safety_margin_tokens": int(safety),
    }
    save_json_safe(SETTINGS_PATH, s)
    return "✅ 저장됨: config/settings.json"

def load_settings_ui():
    s = load_settings()
    return s["max_input_tokens"], s["max_output_tokens"], s["safety_margin_tokens"], "🔄 불러오기 완료"

# =========================
# Gradio UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("## 멀티 모델 챗봇")

    # 전역 상태
    state_history = gr.State([])
    # 설정 상태(탭1에서 바로 반영)
    state_max_in = gr.State(DEFAULT_SETTINGS["max_input_tokens"])
    state_max_out = gr.State(DEFAULT_SETTINGS["max_output_tokens"])
    state_safety = gr.State(DEFAULT_SETTINGS["safety_margin_tokens"])

    with gr.Tabs():
        # ------------------ 탭1: Chat ------------------
        with gr.Tab("💬Chat"):
            with gr.Row():
                provider = gr.Dropdown(
                    choices=["OpenAI ChatGPT", "Google Gemini", "Local Ollama (Gemma 4.0)"],
                    value="Local Ollama (Gemma 4.0)",
                    label="Provider",
                    scale=9
                )
                # 초기화 버튼
                reset_btn = gr.Button("🔄 초기화", scale=1)

            chatbot = gr.Chatbot(type="messages", height=640, label="Chat", show_label=False)

            # --- 입력창 박스 ---
            with gr.Group(): 
                with gr.Row(equal_height=True):
                    msg = gr.Textbox(
                        placeholder="메시지를 입력하세요", 
                        show_label=False, # 높이 정렬을 위해 라벨은 숨김
                        scale=9,
                        container=False # 박스 테두리 중첩 방지
                    )
                    submit_btn = gr.Button(
                        "▶보내기", 
                        scale=1, 
                        variant="primary",
                        min_width=100
                    )

            # 초기 로드: history/설정 불러오기
            def _on_load():
                ui, hist, max_in, max_out, safety = init_chat()
                return ui, hist, max_in, max_out, safety
            demo.load(
                fn=_on_load,
                inputs=None,
                outputs=[chatbot, state_history, state_max_in, state_max_out, state_safety]
            )

            # 전송
            msg.submit( #엔터키입력
                on_submit,
                inputs=[msg, state_history, provider, state_max_in, state_max_out, state_safety],
                outputs=[chatbot, state_history]
            )
            msg.submit(lambda: "", None, msg)

            submit_btn.click( #버튼입력
                on_submit,
                inputs=[msg, state_history, provider, state_max_in, state_max_out, state_safety],
                outputs=[chatbot, state_history]
            )
            submit_btn.click(lambda: "", None, msg)

            # 초기화 동작
            reset_btn.click(
                fn=reset_chat,
                inputs=None,
                outputs=[chatbot, state_history]
            )

            # 내보내기 버튼(탭1)
            with gr.Row():
                with gr.Column(scale=8):
                    with gr.Row():
                        include_system_chk = gr.Checkbox(label="system 메시지 포함", value=False)
                        mode_radio = gr.Radio(choices=["block", "flat"], value="block", label="형식")
                
                # '대화 내역 내보내기'를 체크박스 오른쪽 열에 작게 배치
                export_btn1 = gr.Button("💾 텍스트 파일 저장", scale=2)

            # 결과 파일 1개
            transcript_file1 = gr.File(label="chat_transcript.txt", interactive=False)

            # 버튼 핸들러
            def on_export(history, include_system, mode):
                # export_chat_transcript(history, include_system=False, mode="flat") 가 정의되어 있어야 함
                path = export_chat(history, include_system=include_system, mode=mode)
                return path

            export_btn1.click(
                on_export,
                inputs=[state_history, include_system_chk, mode_radio],
                outputs=[transcript_file1]
            )

        # ------------------ 탭2: Tokens ------------------
        with gr.Tab("🔐Token Authorization"):
            gr.Markdown("OpenAI/Gemini API 키와 기본 모델을 JSON으로 저장/불러옵니다. **키는 UI에 다시 표시하지 않습니다.**")
            with gr.Row():
                openai_key_in = gr.Textbox(label="OpenAI API Key", type="password", placeholder="sk-...", value="")
                openai_model_in = gr.Textbox(label="OpenAI 기본 모델", value=OPENAI_DEFAULT_MODEL)
            with gr.Row():
                gemini_key_in = gr.Textbox(label="Google Gemini API Key", type="password", placeholder="AIza...", value="")
                gemini_model_in = gr.Textbox(label="Gemini 기본 모델", value=GEMINI_DEFAULT_MODEL)
            with gr.Row():
                ollama_url_in = gr.Textbox(label="Ollama Base URL", value=OLLAMA_BASE_URL)
                ollama_model_in = gr.Textbox(label="Ollama 기본 모델", value=OLLAMA_DEFAULT_MODEL)

            with gr.Row():
                save_keys_btn = gr.Button("🔐 토큰/모델 저장 → keys.json")
                load_keys_btn = gr.Button("📥 모델 불러오기(키는 미표시)")

            token_status = gr.Markdown("")

            save_keys_btn.click(
                save_keys,
                inputs=[openai_key_in, openai_model_in, gemini_key_in, gemini_model_in, ollama_url_in, ollama_model_in],
                outputs=[token_status]
            )
            load_keys_btn.click(
                load_keys_ui,
                inputs=None,
                outputs=[openai_model_in, gemini_model_in, ollama_url_in, ollama_model_in, token_status]
            )

        # ------------------ 탭3: Settings ------------------
        with gr.Tab("⚙️Model Settings"):
            gr.Markdown("최대 입력/출력 토큰, 세이프티 마진을 JSON으로 저장/불러옵니다. (탭1에서 즉시 반영)")
            with gr.Row():
                max_in_box = gr.Number(label="최대 입력 토큰 (예: 6000)", value=DEFAULT_SETTINGS["max_input_tokens"], precision=0)
                max_out_box = gr.Number(label="최대 출력 토큰 (예: 512)", value=DEFAULT_SETTINGS["max_output_tokens"], precision=0)
                safety_box = gr.Number(label="세이프 마진 (예: 256)", value=DEFAULT_SETTINGS["safety_margin_tokens"], precision=0)

            with gr.Row():
                save_settings_btn = gr.Button("🧭 설정 저장 → settings.json")
                load_settings_btn = gr.Button("📥 설정 불러오기")

            settings_status = gr.Markdown("")

            # 저장: 파일 저장 + 탭1 사용 상태(state)도 갱신
            def on_save_settings(max_in, max_out, safety):
                msg = save_settings(max_in, max_out, safety)
                return int(max_in), int(max_out), int(safety), msg

            save_settings_btn.click(
                on_save_settings,
                inputs=[max_in_box, max_out_box, safety_box],
                outputs=[state_max_in, state_max_out, state_safety, settings_status]
            )

            # 불러오기: UI 입력칸/상태 모두 갱신
            def on_load_settings_full():
                mi, mo, sa, note = load_settings_ui()
                return mi, mo, sa, mi, mo, sa, note

            load_settings_btn.click(
                on_load_settings_full,
                inputs=None,
                outputs=[state_max_in, state_max_out, state_safety, max_in_box, max_out_box, safety_box, settings_status]
            )
        # ------------------ 탭4: System Prompt ------------------
        with gr.Tab("📝System Prompt Editor"):
            gr.Markdown("### 시스템 프롬프트 및 시작 프롬프트 편집")
            gr.Markdown("- 시스템 프롬프트: `assets/prompt/system-message.txt`\n- 시작 프롬프트(초기 assistant 메시지): `assets/prompt/prompt-message.txt`")

            with gr.Row():
                sys_prompt_box = gr.Textbox(
                    lines=14, label="시스템 프롬프트 (system-message.txt)",
                    placeholder="📂 먼저 불러오기를 눌러 현재 프롬프트를 확인하세요."
                )
                start_prompt_box = gr.Textbox(
                    lines=14, label="시작 프롬프트 (prompt-message.txt)",
                    placeholder="📂 먼저 불러오기를 눌러 현재 프롬프트를 확인하세요."
                )

            with gr.Row():
                load_prompts_btn = gr.Button("📂 프롬프트 불러오기")
                save_sys_btn = gr.Button("💾 시스템 프롬프트 저장")
                save_start_btn = gr.Button("💾 시작 프롬프트 저장")
                apply_and_reset_btn = gr.Button("✅ 변경 적용 후 대화 초기화")
            
            gr.Markdown("⚠️ 프롬프트를 갱신하면 진행중인 대화기록은 모두 초기화됩니다.\n" \
                    "원치 않을 경우 먼저 저장하시길 바랍니다.\n")

            prompt_status = gr.Markdown("")

            # --- 파일 경로 상수 ---
            _SYS_PATH = "assets/prompt/system-message.txt"
            _START_PATH = "assets/prompt/prompt-message.txt"

            # --- 파일 유틸 ---
            def _read_text_safe(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return f.read()
                except FileNotFoundError:
                    # 디렉토리 없을 수 있으니 만들어 두기
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    return ""
                except Exception as e:
                    return f"(오류) {e}"

            def _write_text_safe(path, text):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)

            # --- 로드/세이브 함수들 ---
            def load_both_prompts():
                sys_txt = _read_text_safe(_SYS_PATH)
                start_txt = _read_text_safe(_START_PATH)
                note = "📂 프롬프트 불러오기 완료"
                return sys_txt, start_txt, note

            def save_system_prompt(text):
                _write_text_safe(_SYS_PATH, text or "")
                return "💾 시스템 프롬프트 저장 완료"

            def save_start_prompt(text):
                _write_text_safe(_START_PATH, text or "")
                return "💾 시작 프롬프트 저장 완료"

            # 변경 적용: 파일 저장 이후 init_chat() 로직을 활용해
            # 시스템/시작 프롬프트를 읽어 새 history를 구성하고, 탭1의 chatbot과 state_history를 갱신
            def apply_prompts_and_reset():
                # 새 system/prompt 파일 저장은 이미 다른 버튼으로 했다고 가정
                ui, hist = reset_chat()
                # 경고 + 초기화 완료 메시지
                note = "✅ 변경 사항이 적용되어 대화가 초기화되었습니다."
                return ui, hist, note

            # --- 버튼 연결 ---
            load_prompts_btn.click(
                load_both_prompts,
                inputs=None,
                outputs=[sys_prompt_box, start_prompt_box, prompt_status]
            )

            save_sys_btn.click(
                save_system_prompt,
                inputs=[sys_prompt_box],
                outputs=[prompt_status]
            )

            save_start_btn.click(
                save_start_prompt,
                inputs=[start_prompt_box],
                outputs=[prompt_status]
            )

            apply_and_reset_btn.click(
                apply_prompts_and_reset,
                inputs=None,
                outputs=[chatbot, state_history, prompt_status]
            )

# demo.launch(share=True)
demo.launch(server_name="0.0.0.0", auth=("1","1"))