class ContextManager:
    # 전역 상수 설정
    USER_NAME = "User"
    BOT_NAME = "Tanjiro"

    def __init__(self):
        self.user_name = self.USER_NAME
        self.bot_name = self.BOT_NAME
        self.history = []

    def getUserName(self) -> str:
        return self.user_name
    
    def getBotName(self) -> str:
        return self.bot_name
    
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