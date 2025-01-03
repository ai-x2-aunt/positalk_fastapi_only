from style_model_handler import create_generator, convert_with_ai
from formal_converter import init_formal_generator

def init_ai():
    create_generator()
    init_formal_generator()

def convert_text(text: str, style: str) -> str:
    if style == "formal":
        try:
            return convert_with_ai(style, text)
        except Exception as e:
            print(f"AI 변환 중 오류 발생: {e}")
    
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