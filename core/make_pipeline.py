from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import os

class MakePipeline:
    # 모델명
    MODEL_ID = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
    
    # 변수초기화
    # model_id
    # tokenizer
    # llm
    def __init__(self, model_id: str = MODEL_ID):
        print("[torch] is available:", torch.cuda.is_available())
        print("[device] default:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_id = model_id
        self.tokenizer = None
        self.llm = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 불러오기
    def build(self, type: str):
        if(type == 'hf'):
            # 허깅 페이스 secret에 등록된 토큰 로드
            access_token = os.environ.get("HF_TOKEN")
        else:
            # 로컬 실행시 token.txt에서 토큰 로드
            with open("token.txt", "r") as f:
                access_token = f.read().strip()

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=access_token)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, token=access_token)
        self.tokenizer = tokenizer

        # 허깅 페이스 업로드 시 f16 사용 안 함
        if(type == 'hf'):
            llm = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )

        else:
            model.eval()
            llm = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16
            )
            if torch.cuda.is_available():
                model.to("cuda")

        self.llm = llm
    
    # 모델 출력 생성 함수
    def character_chat(self, prompt):
        print("[debug] generating...")
        outputs = self.llm(
            prompt,
            do_sample=True,
            max_new_tokens=96,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=True
        )
        full_text = outputs[0]['generated_text']
        return full_text