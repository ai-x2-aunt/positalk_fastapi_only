from ai_model import create_generator, convert_with_ai

_generator = None

def init_ai():
    global _generator
    _generator = create_generator()

def convert_text(text: str, style: str) -> str:
    if _generator:
        try:
            return convert_with_ai(_generator, style, text)
        except Exception as e:
            print(f"AI 변환 중 오류 발생: {e}")
    
    # AI 변환 실패 또는 AI 사용 불가능할 때 기본 변환 로직 사용
    if style == "pretty":
        return f"{text}에용~"
    elif style == "cute":
        return f"{text}~♥"
    elif style == "polite":
        return f"{text}입니다"
    elif style == "formal":
        return f"{text}하십니다"
    elif style == "friendly":
        return f"{text}야"
    return text 