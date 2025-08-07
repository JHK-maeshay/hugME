import gradio as gr
from core.context_manager import ContextManager
from core.make_pipeline import MakePipeline
from core.make_reply import generate_reply
from core.utils import load_config as load_full_config, save_config as save_full_config, load_llm_config
import re

def create_interface(ctx: ContextManager, makePipeline: MakePipeline):
    with gr.Blocks(css="""
    .chat-box { max-height: 500px; overflow-y: auto; padding: 10px; border: 1px solid #ccc; border-radius: 10px; }
    .bubble-left { background-color: #f1f0f0; border-radius: 10px; padding: 10px; margin: 5px; max-width: 70%; float: left; clear: both; }
    .bubble-right { background-color: #d1e7ff; border-radius: 10px; padding: 10px; margin: 5px; max-width: 70%; float: right; clear: both; text-align: right; }
    .reset-btn-container { text-align: right; margin-bottom: 10px; }
    """) as demo:
        with gr.Tabs():
            ### 1. 채팅 탭 ###
            with gr.TabItem("💬 탄지로와 대화하기"):

                with gr.Column():
                    with gr.Row():
                        gr.Markdown("### 탄지로와 대화하기")
                        reset_btn = gr.Button("🔁 대화 초기화", elem_classes="reset-btn-container", scale=0.25)
                    chat_output = gr.HTML(elem_id="chat-box")
                    user_input = gr.Textbox(label="메시지 입력", placeholder="탄지로에게 말을 걸어보세요")
                    state = gr.State(ctx)

            # history 읽어서 화면에 뿌리는 역할
            def render_chat(ctx: ContextManager):

                def parse_emotion_text(text: str) -> str:
                    """
                    *...* 부분은 회색 텍스트로 바꾸고, 줄바꿈을 추가하여 HTML로 반환
                    """
                    segments = []
                    pattern = re.compile(r"\*(.+?)\*|([^\*]+)")
                    matches = pattern.findall(text)

                    for action, plain in matches:
                        if action:
                            segments.append(f"<div style='color:gray'>*{action}*</div>")
                        elif plain:
                            for line in plain.strip().splitlines():
                                line = line.strip()
                                if line:
                                    segments.append(f"<div>{line}</div>")
                    return "\n".join(segments)

                html = ""
                for item in ctx.getHistory():
                    parsed = parse_emotion_text(item['text'])
                    if item["role"] == "user":
                        html += f"<div class='bubble-right'>{parsed}</div>"
                    elif item["role"] == "bot":
                        html += f"<div class='bubble-left'>{parsed}</div>"

                return gr.update(value=html)

            def on_submit(user_msg: str, ctx: ContextManager):
                # 사용자 입력 history에 추가
                ctx.addHistory("user", user_msg)

                # 사용자 입력을 포함한 채팅 우선 렌더링
                html = render_chat(ctx)
                yield html, "", ctx

                # 봇 응답 생성
                generate_reply(ctx, makePipeline, user_msg)

                # 응답을 포함한 전체 history 기반 렌더링
                html = render_chat(ctx)
                yield html, "", ctx

            # history 초기화
            def reset_chat(ctx: ContextManager):
                ctx.clearHistory()
                return gr.update(value=""), "", ctx

            user_input.submit(on_submit, inputs=[user_input, state], outputs=[chat_output, user_input, state], queue=True)
            reset_btn.click(reset_chat, inputs=[state], outputs=[chat_output, user_input, state])

            ### 2. 설정 탭 ###
            with gr.TabItem("⚙️ 모델 설정"):
                gr.Markdown("### LLM 파라미터 설정")

                with gr.Row():
                    temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                    repetition_penalty = gr.Slider(0.8, 2.0, value=1.05, step=0.01, label="Repetition Penalty")

                with gr.Row():
                    max_tokens = gr.Slider(16, 2048, value=96, step=8, label="Max New Tokens")

                apply_btn = gr.Button("✅ 설정 적용")

                def update_config(temp, topp, max_tok, repeat):
                    makePipeline.update_config({
                        "temperature": temp,
                        "top_p": topp,
                        "max_new_tokens": max_tok,
                        "repetition_penalty": repeat
                    })
                    return gr.update(value="✅ 설정 적용 완료")

                # 🔻 설정 불러오기 / 내보내기 버튼들
                with gr.Row():
                    load_btn = gr.Button("📂 설정 불러오기")
                    save_btn = gr.Button("💾 설정 내보내기")

                def load_config():
                    llm_cfg = load_llm_config("config.json")
                    return (
                        llm_cfg.get("temperature", 0.7),
                        llm_cfg.get("top_p", 0.9),
                        llm_cfg.get("repetition_penalty", 1.05),
                        llm_cfg.get("max_new_tokens", 96),
                        "📂 설정 불러오기 완료"
                    )

                def save_config(temp, topp, repeat, max_tok):
                    # 기존 전체 설정 불러오기
                    config = load_full_config("config.json")

                    # LLM 블록만 새로 대입
                    config["llm"] = {
                        "temperature": temp,
                        "top_p": topp,
                        "repetition_penalty": repeat,
                        "max_new_tokens": max_tok
                    }

                    # 전체 저장
                    save_full_config(config, path="config.json")

                    return gr.update(value="💾 설정 저장 완료")
                
                # ✅ 맨 아래에 상태창 배치
                status = gr.Textbox(label="", interactive=False)

                # 📂 버튼 동작 연결
                apply_btn.click(
                    update_config,
                    inputs=[temperature, top_p, max_tokens, repetition_penalty],
                    outputs=[status]  # 혹은 []
                )
                
                load_btn.click(
                    load_config,
                    inputs=None,
                    outputs=[temperature, top_p, repetition_penalty, max_tokens, status]
                )

                save_btn.click(
                    save_config,
                    inputs=[temperature, top_p, repetition_penalty, max_tokens],
                    outputs=[status]
                )

            ### 3. 프롬프트 편집 탭 ###
            with gr.TabItem("📝 프롬프트 설정"):
                gr.Markdown("### 사용자 및 캐릭터 이름 설정")

                with gr.Row():
                    user_name = gr.Textbox(label="👤 사용자 이름")
                    bot_name = gr.Textbox(label="🤖 캐릭터 이름")

                name_status = gr.Textbox(label="", interactive=False)

                with gr.Row():
                    load_name_btn = gr.Button("📂 이름 불러오기")
                    save_name_btn = gr.Button("💾 이름 저장하기")

                def load_names():
                    cha_cfg = load_full_config("config.json").get("cha", {})
                    return (
                        cha_cfg.get("user_name", "user"),
                        cha_cfg.get("bot_name", "Tanjiro"),
                        "📂 이름 불러오기 완료"
                    )

                def save_names(user, bot):
                    config = load_full_config("config.json")
                    config["cha"] = {
                        "user_name": user,
                        "bot_name": bot
                    }
                    save_full_config(config, path="config.json")

                    ctx.setUserName(user)
                    ctx.setBotName = (bot)

                    return "💾 이름 저장 완료!"

                load_name_btn.click(
                    load_names,
                    inputs=None,
                    outputs=[user_name, bot_name, name_status]
                )

                save_name_btn.click(
                    save_names,
                    inputs=[user_name, bot_name],
                    outputs=[name_status]
                )

                #초기화 시점에서 이름 한번 불러오기
                demo.load(
                    fn=load_names,
                    inputs=None,
                    outputs=[user_name, bot_name, name_status]
                )

                gr.Markdown("### 캐릭터 및 세계관 프롬프트 편집")

                prompt_editor = gr.Textbox(
                    lines=20,
                    label="텍스트 (init.txt)",
                    placeholder="!! 반드시 불러오기를 먼저 하세요 !!",
                    interactive=True
                )
                with gr.Row():
                    gr.Markdown("#### !! 반드시 불러오기를 먼저 하세요 !!")

                with gr.Row():
                    load_prompt_btn = gr.Button("📂 현재 프롬프트 불러오기")
                    save_prompt_btn = gr.Button("💾 작성한 프롬프트로 교체")

                def load_prompt():
                    try:
                        with open("assets/prompt/init.txt", "r", encoding="utf-8") as f:
                            return f.read()
                    except FileNotFoundError:
                        return ""

                def save_prompt(text):
                    with open("assets/prompt/init.txt", "w", encoding="utf-8") as f:
                        f.write(text)
                    return "💾 저장 완료!"

                load_prompt_btn.click(
                    load_prompt,
                    inputs=None,
                    outputs=prompt_editor
                )

                save_prompt_btn.click(
                    save_prompt,
                    inputs=[prompt_editor],
                    outputs=[save_prompt_btn]
                )

        return demo