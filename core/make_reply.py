import re

# 생성된 모든 봇 응답 기록
def generate_reply(ctx, makePipeLine, user_msg):
    # 최초 응답
    response = generate_valid_response(ctx, makePipeLine, user_msg)
    ctx.addHistory("bot", response)

    # 불안정한 응답이 유도되므로 사용하지 않음
    '''
    # 응답이 끊겼다면 추가 생성
    if is_truncated_response(response):
        continuation = generate_valid_response(ctx, makePipeLine, response)
        ctx.addHistory("bot", continuation)
    '''

# 봇 응답 1회 생성
def generate_valid_response(ctx, makePipeline, user_msg) -> str:
    user_name = ctx.getUserName()
    bot_name = ctx.getBotName()

    while True:
        prompt = build_prompt(ctx.getHistory(), user_msg, user_name, bot_name)
        full_text = makePipeline.character_chat(prompt)
        response = extract_response(full_text)
        print(f"debug: {response}")
        if is_valid_response(response, user_name, bot_name):
            break
    return clean_response(response, bot_name)

# 입력 프롬프트 정리
def build_prompt(history, user_msg, user_name, bot_name):
    with open("assets/prompt/init.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    # 최근 대화 히스토리를 일반 텍스트로 재구성
    dialogue = ""
    for turn in history[-16:]:
        role = user_name if turn["role"] == "user" else bot_name
        dialogue += f"{role}: {turn['text']}\n"

    dialogue += f"{user_name}: {user_msg}\n"

    # 모델에 맞는 포맷 구성
    prompt = f"""### Instruction:
{system_prompt}

{dialogue}
### Response:
{bot_name}:"""
    return prompt

# 출력에서 응답 추출 (HyperCLOVAX 포맷에 맞게)
def extract_response(full_text):
    # '### Response:' 이후 텍스트 추출
    if "### Response:" in full_text:
        reply = full_text.split("### Response:")[-1].strip()
    else:
        reply = full_text.strip()
    return reply

# 응답 유효성 검사
def is_valid_response(text: str, user_name, bot_name) -> bool:
    if user_name + ":" in text:
        return False
    return True

# 출력 정제
def clean_response(text: str, bot_name):
    # bot_name 제거
    text = re.sub(rf"{bot_name}:\s*", "", text).strip()
    # 미완성 문장 제거
    return clean_truncated_response(text)

# 미완성 문장 삭제
def clean_truncated_response(text: str) -> str:
    """
    응답 텍스트가 미완성된 문장으로 끝나면 마지막 문장을 제거하여 반환,
    그렇지 않으면 원문 그대로 반환.
    """

    # 문장 분리 ('.', '?', '!', '~' 등 기준 + 줄바꿈 포함)
    sentence_end_pattern = r"(?<=[\.?!~])\s|\n"
    segments = re.split(sentence_end_pattern, text.strip())

    if not segments:
        return text.strip()

    cleaned = []
    for s in segments:
        s = s.strip()
        if not s:
            continue
        # 문장 부호로 끝나는 경우만 포함
        if re.search(r"[.?!~…\u2026\u2639\u263A\u2764\uD83D\uDE0A\uD83D\uDE22]$", s):
            cleaned.append(s)
        else:
            break  # 불완전한 문장이므로 이후 모두 제거

    # 만약 모든 문장이 끝맺음을 잘 했다면 → 원문 반환
    result = " ".join(cleaned)
    return result if result != "" else text.strip()