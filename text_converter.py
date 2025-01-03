# from utils.ai_model import convert_with_ai

def convert_text(text: str, style: str) -> str:
    # try:
        # return convert_with_ai(style, text)
        
    # except Exception as e:
        # print(f"Error in text conversion: {e}")

        # 에러 발생시 기본 변환 로직 사용
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