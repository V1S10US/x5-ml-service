import random
import time
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from schemas import ScoringRequest, ScoringResponse, ScoredCandidate
from custom_logger import logger

# Имитация инициализации моделей при старте
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML models... (Stub mode)")
    # Здесь в будущем будет загрузка весов (torch.load...)
    logger.info("ML models loaded successfully")
    yield
    logger.info("Shutting down ML service")

app = FastAPI(title="Candidate Scoring Service", version="0.1.0", lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {"status": "healthy", "service": "ml_scorer"}

@app.post("/score_candidates", response_model=ScoringResponse)
async def score_candidates(payload: ScoringRequest):
    """
    Принимает вакансию и список кандидатов.
    Возвращает список ID кандидатов и их скор (0-1).
    """
    req_id = f"req_{int(time.time())}"
    vacancy_title = payload.vacancy.title
    candidate_count = len(payload.candidates)
    
    logger.info(f"[{req_id}] Received batch scoring request. Vacancy: '{vacancy_title}', Candidates: {candidate_count}")

    results = []
    
    # Эмуляция процессинга
    try:
        for candidate in payload.candidates:
            # --- ZAGLUSHKA (STUB) ---
            # Здесь будет вызов реальной модели: 
            # score = model.predict(payload.vacancy.description, candidate.cv_text)
            
            # Пока возвращаем random
            random_score = round(random.uniform(0.1, 0.99), 2)
            # ------------------------
            
            results.append(ScoredCandidate(
                candidate_id=candidate.id,
                score=random_score
            ))
            
            # Логируем детали для отладки (в продакшене уровень DEBUG)
            logger.info(f"[{req_id}] Scored candidate {candidate.id}: {random_score}")

    except Exception as e:
        logger.error(f"[{req_id}] Error processing batch: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")

    logger.info(f"[{req_id}] Batch processing completed.")
    
    return ScoringResponse(
        vacancy_title=vacancy_title,
        results=results
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

