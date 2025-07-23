import gradio as gr
from core.context_manager import ContextManager
from core.make_pipeline import MakePipeline
from core.make_reply import generate_reply

def create_interface(ctx: ContextManager, makePipeline: MakePipeline):
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
            state = gr.State(ctx)

        # history 읽어서 화면에 뿌리는 역할
        def render_chat(ctx: ContextManager):
            html = ""
            for item in ctx.getHistory():
                if item["role"] == ctx.getUserName():
                    html += f"<div class='bubble-right'>{item['text']}</div>"
                elif item["role"] == ctx.getBotName():
                    html += f"<div class='bubble-left'>{item['text']}</div>"
            return gr.update(value=html)

        def on_submit(user_msg: str, ctx: ContextManager):
            # 사용자 메세지 추가
            ctx.addMessage("user", user_msg)

            # 유저 답변을 포함한 HTML 렌더링
            html = render_chat(ctx)
            yield html, "", ctx.getHistory()

            # 봇 응답 생성
            generate_reply(ctx, makePipeline, user_msg)

            # 봇 답변을 포함한 HTML 렌더링
            html = render_chat(ctx)
            yield html, "", ctx.getHistory()

        # history 초기화
        def reset_chat():
            ctx.resetHistory()
            return gr.update(value=""), "", ctx.getHistory()

        user_input.submit(on_submit, inputs=[user_input, state], outputs=[chat_output, user_input, state], queue=True)
        reset_btn.click(reset_chat, inputs=None, outputs=[chat_output, user_input, state])

        return demo