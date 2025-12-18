import time
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from schemas import ScoringRequest, ScoringResponse, ScoredCandidateResult, ScoredCandidateResultLLM, ScoringResponseLLM
from custom_logger import logger
# Импортируем наш новый класс
from model_engine import lr_scorer
from llm_engine import llm_scorer

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- LIFESPAN START ---")
    
    # 1. Обучение модели при старте
    try:
        logger.info("Generating dummy data and training Linear Regression model...")
        lr_scorer.train()
        logger.info("Model artifact created in memory.")
    except Exception as e:
        logger.error(f"Failed to train model on startup: {e}")
        raise e
        
    logger.info("Service is ready to accept requests.")
    yield
    logger.info("--- LIFESPAN SHUTDOWN ---")

# --- App Definition ---
app = FastAPI(
    title="Candidate Scoring Service",
    description="Hybrid scoring system: TF-IDF Baseline + LLM Analysis",
    version="0.5.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml_scorer"}

@app.post("/score_candidates", response_model=ScoringResponse)
async def score_candidates(payload: ScoringRequest):
    """
    Основной эндпоинт скоринга.
    Принимает: VacancyData, List[CandidateFeatures]
    Возвращает: List[ScoredCandidateResult] с детализацией скоров.
    """
    req_id = f"req_{int(time.time())}"
    
    vacancy_title = payload.vacancy.title
    vacancy_desc = payload.vacancy.description
    
    logger.info(f"[{req_id}] Processing batch via LR Model for: '{vacancy_title}'")

    # Конвертируем кандидатов в список словарей
    candidates_dicts = [c.model_dump() for c in payload.candidates]

    try:
        # Вызываем метод предсказания у обученной модели
        # Теперь мы передаем title и desc отдельно, конкатенация внутри
        scores = lr_scorer.predict(vacancy_title, vacancy_desc, candidates_dicts)
        
        results = []
        for cand_obj, score in zip(payload.candidates, scores):
            results.append(ScoredCandidateResult(
                candidate_id=cand_obj.id,
                total_score=round(score, 4), # Округляем до 4 знаков
                
                # Для этого пайплайна text_score и total_score - это одно и то же,
                # так как модель учится на всем сразу.
                text_score=round(score, 4),
                meta_score=0.0 # В данной модели мета-признаки "размазаны" в тексте
            ))
            logger.info(f"[{req_id}] CandID: {cand_obj.id} -> Predicted: {score:.4f}")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return ScoringResponse(
        vacancy_title=vacancy_title,
        processed_count=len(results),
        results=results
    )

@app.post("/score_candidates_llm", response_model=ScoringResponseLLM)
async def score_candidates_llm(payload: ScoringRequest):
    """
    УМНЫЙ метод (LLM).
    Использует GPT/LLM для анализа.
    Время ответа: ~1-5 sec на кандидата.
    Возвращает 'reason' (обоснование).
    """
    req_id = f"req_llm_{int(time.time())}"
    vacancy_title = payload.vacancy.title
    
    logger.info(f"[{req_id}] LLM scoring request for '{vacancy_title}' ({len(payload.candidates)} cands)")

    results = []
    
    # Обрабатываем кандидатов
    # В проде здесь стоит использовать asyncio.gather для параллелизма (с лимитом семафора)
    # Но сейчас сделаем простой цикл
    for cand in payload.candidates:
        try:
            cand_dict = cand.model_dump()
            
            # Вызов LLM движка
            llm_result = await llm_scorer.predict_one(
                vacancy_title,
                payload.vacancy.description or "",
                cand_dict
            )
            
            score = llm_result.get("score", 0.0)
            reason = llm_result.get("reason", "No reason provided")
            
            # Логируем результат
            logger.info(f"[{req_id}] Cand {cand.id}: {score} | {reason[:30]}...")

            results.append(ScoredCandidateResultLLM(
                candidate_id=cand.id,
                total_score=score,
                text_score=score,
                meta_score=0.0,
                reason=reason
            ))
            
        except Exception as e:
            logger.error(f"[{req_id}] Failed for cand {cand.id}: {e}")
            # Возвращаем ошибку в скоре, но не валим весь батч
            results.append(ScoredCandidateResultLLM(
                candidate_id=cand.id,
                total_score=0.0,
                text_score=0.0,
                meta_score=0.0,
                reason=f"Processing Error: {str(e)}"
            ))

    return ScoringResponseLLM(
        vacancy_title=vacancy_title,
        processed_count=len(results),
        results=results
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
