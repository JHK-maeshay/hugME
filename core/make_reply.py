import re

from core.make_pipeline import MakePipeline

# 생성된 모든 봇 응답 기록
def generate_reply(ctx, makePipeLine: MakePipeline, user_msg):

    # 최초 응답
    response = generate_valid_response(ctx, makePipeLine, user_msg)
    ctx.addHistory("bot", response)

    # 응답이 끊겼다면 추가 생성
    if is_truncated_response(response):
        continuation = generate_valid_response(ctx, makePipeLine, response)
        ctx.addHistory("bot", continuation)

# 봇 응답 1회 생성
def generate_valid_response(ctx, makePipeline: MakePipeline, user_msg: str) -> str:
    user_name = ctx.getUserName()
    bot_name = ctx.getBotName()

    while True:
        prompt = build_prompt(ctx.getHistory(), user_msg, user_name, bot_name)
        full_text = makePipeline.character_chat(prompt)
        response = extract_response(full_text, prompt, user_name, bot_name)
        if is_valid_response(response, user_name, bot_name):
            break
    return clean_response(response, bot_name)

# 입력 프롬프트 정리
def build_prompt(history, user_msg, user_name, bot_name):
    with open("assets/prompt/init.txt", "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    for turn in history[-16:]:
        role = user_name if turn["role"] == "user" else bot_name
        prompt += f"{role}: {turn['text']}\n"

    prompt += f"{user_name}: {user_msg}\n"
    prompt += f"{bot_name}:"
    return prompt

# 출력에서 응답 추출
def extract_response(full_text, prompt, user_name, bot_name):
    if full_text.startswith(prompt):
        reply = full_text[len(prompt):].strip()
    else:
        reply = full_text.split(f"{bot_name}:")[-1].strip()
    user_token = f"\n{user_name}:"
    if user_token in reply:
        reply = reply.split(user_token)[0].strip()
    return reply

# 응답 유효성 검사
def is_valid_response(text: str, user_name, bot_name) -> bool:
    if user_name + ":" in text:
        return False
    if bot_name + ":" in text:
        return False
    return True

# 응답 형식 정리
def clean_response(text: str, bot_name):
    return re.sub(rf"{bot_name}:\\s*", "", text).strip()

# 중단된 응답 여부 검사
def is_truncated_response(text: str) -> bool:
    return re.search(r"[.?!…\u2026\u2639\u263A\u2764\uD83D\uDE0A\uD83D\uDE22]$", text.strip()) is None