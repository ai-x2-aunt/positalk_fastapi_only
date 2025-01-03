from transformers import pipeline
import torch
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from formal_converter import convert_to_formal

MODEL_NAME = "heegyu/kobart-text-style-transfer"
TIMEOUT_SECONDS = 10
_generator = None

def create_generator():
    print("AI 모델 초기화 중...")
    global _generator
    _generator = pipeline(
        "text2text-generation",
        model=MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs={"torch_dtype": torch.float16},
        truncation=True
    )

def convert_with_ai(style: str, text: str) -> str:
    if style == "formal":
        return convert_to_formal(text)
        
    style_prompts = {
        "pretty": f"일반체를 귀여운체로: {text}",
        "cute": f"일반체를 애교체로: {text}",
        "polite": f"일반체를 존댓말로: {text}",
        "friendly": f"일반체를 친근체로: {text}"
    }
    
    try:
        print(f"텍스트 변환 시작: {text[:20]}...")
        start_time = time.time()
        
        prompt = style_prompts.get(style, f"변환: {text}")
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_generator, 
                prompt,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                truncation=True
            )
            
            try:
                result = future.result(timeout=TIMEOUT_SECONDS)[0]["generated_text"]
                
                elapsed_time = time.time() - start_time
                print(f"변환 완료 (소요시간: {elapsed_time:.2f}초)")
                return result.strip()
                
            except TimeoutError:
                print(f"시간 초과 ({TIMEOUT_SECONDS}초 초과)")
                executor.shutdown(wait=False, cancel_futures=True)
                raise Exception("변환 시간이 너무 오래 걸립니다")
        
    except Exception as e:
        print(f"변환 실패: {str(e)}")
        raise Exception(f"AI 변환 중 오류 발생: {str(e)}")