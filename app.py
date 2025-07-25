import gradio as gr

from core.make_pipeline import MakePipeline
from core.context_manager import ContextManager
from core.launch_gradio import create_interface

###########################
# Upload to Huggling Face #
###########################

# 모델 불러오기
makePipeline = MakePipeline()
makePipeline.build("hf")

# 채팅 기록 관리자
ctx = ContextManager()

# Gradio 인터페이스 시작
demo = create_interface(ctx, makePipeline)
demo.launch()
    
