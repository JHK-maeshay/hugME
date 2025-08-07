from core.utils import load_cha_config

class ContextManager:
    def __init__(self):
        cha_cfg = load_cha_config("config.json")
        self.user_name = cha_cfg.get("user_name", "user")
        self.bot_name = cha_cfg.get("bot_name", "Tanjiro")
        self.history = []

    def getUserName(self) -> str:
        return self.user_name
    
    def setUserName(self, user_name):
        self.user_name = user_name
    
    def getBotName(self) -> str:
        return self.bot_name
    
    def setBotName(self, bot_name):
        self.bot_name = bot_name
    
    def getHistory(self) -> str:
        return self.history
    
    def setHistory(self, new_history: list):
        self.history = new_history
    
    # 대화 기록을 history에 추가
    def addHistory(self, role: str, text: str):
        self.history.append({"role": role, "text": text})

    # 대화 기록 초기화
    def clearHistory(self):
        self.history = []