from transformers import pipeline
import torch
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

MODEL_NAME = "9unu/formal_speech_translation"
TIMEOUT_SECONDS = 10

def init_formal_generator():
    global _generator
    if not _generator:
        print("격식체 변환 모델 초기화 중...")
        _generator = pipeline(
            "text2text-generation",
            model=MODEL_NAME,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs={"torch_dtype": torch.float16},
            truncation=True
        )

def _generate_with_timeout(text: str) -> str:
    prompt = f"비격식체를 아주 격식있게: {text}"
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _generator,
            prompt,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            truncation=True
        )
        
        try:
            return future.result(timeout=TIMEOUT_SECONDS)[0]["generated_text"].strip()
        except TimeoutError:
            executor.shutdown(wait=False, cancel_futures=True)
            raise Exception("변환 시간이 너무 오래 걸립니다")

def convert_to_formal(text: str) -> str:
    if not _generator:
        init_formal_generator()
        
    try:
        print(f"격식체 변환 시작: {text[:20]}...")
        start_time = time.time()
        
        result = _generate_with_timeout(text)
        
        elapsed_time = time.time() - start_time
        print(f"격식체 변환 완료 (소요시간: {elapsed_time:.2f}초)")
        return result
        
    except Exception as e:
        print(f"격식체 변환 실패: {str(e)}")
        raise Exception(f"격식체 변환 중 오류 발생: {str(e)}") 