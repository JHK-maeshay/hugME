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
        self.config = {  # 초기값
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "max_new_tokens": 96
        }

    # 모델 불러오기
    def build(self, type: str):
        if(type == 'ui'):
            print("[build] UI 테스트용 - 모델 로딩 생략")
            return
        
        if(type == 'hf'):
            # 허깅 페이스 secret에 등록된 토큰 로드
            access_token = os.environ.get("HF_TOKEN")
        else:
            # 로컬 실행시 token.txt에서 토큰 로드
            with open("token.txt", "r") as f:
                access_token = f.read().strip()

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=access_token)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, token=access_token, trust_remote_code=True)
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

    # 파리미터 설정
    def update_config(self, new_config: dict):
        self.config.update(new_config)
        print("[config] updated:", self.config)

    # 모델 출력 생성 함수
    def character_chat(self, prompt):
        print("[debug] generating with:", self.config)

        outputs = self.llm(
            prompt,
            do_sample=True,
            max_new_tokens=self.config["max_new_tokens"],
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            repetition_penalty=self.config["repetition_penalty"],
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=True
        )
        return outputs[0]["generated_text"]