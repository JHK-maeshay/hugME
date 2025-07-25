from core.make_pipeline import MakePipeline
from core.context_manager import ContextManager
from core.launch_gradio import create_interface
import argparse

########################
# Start with localhost #
########################
# --testui to test ui  #
########################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testui", action="store_true", help="UI만 실행 여부")
    args = parser.parse_args()

    # 모델 불러오기
    if args.testui:
        makePipeline = MakePipeline()
        makePipeline.build("ui")
    else:
        makePipeline = MakePipeline()
        makePipeline.build("lh")

    # 채팅 기록 관리자
    ctx = ContextManager()

    # Gradio 인터페이스 시작
    demo = create_interface(ctx, makePipeline)
    demo.launch()