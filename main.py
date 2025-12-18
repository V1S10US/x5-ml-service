import time
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

# Импортируем новые имена моделей
from schemas import ScoringRequest, ScoringResponse, ScoredCandidateResult
from custom_logger import logger
from model_engine import scorer_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing ML Scorer Engine...")
    # Здесь можно добавить warmup моделей или подгрузку словарей
    logger.info("ML Engine is ready to accept requests.")
    yield

app = FastAPI(
    title="Candidate Scoring Service", 
    description="Microservice for ranking candidates based on CV text and metadata",
    version="0.3.0", 
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
    
    # Доступ теперь через .title, а не ["название вакансии"]
    vacancy_title = payload.vacancy.title
    vacancy_desc = payload.vacancy.description or "" # Handle None
    
    # Комбинируем текст вакансии для TF-IDF: Title (вес x3) + Description
    vacancy_full_text = (vacancy_title + " ") * 3 + vacancy_desc
    
    count = len(payload.candidates)
    logger.info(f"[{req_id}] Start processing batch. Vacancy: '{vacancy_title}', Candidates: {count}")

    # Конвертируем список Pydantic объектов в список словарей для engine
    # model_dump() выгружает все поля, включая extra (из Excel)
    candidates_dicts = [c.model_dump() for c in payload.candidates]

    try:
        # Вызов обновленного метода predict_batch (который теперь возвращает 3 значения)
        # Нам нужно будет немного доработать model_engine.py, чтобы он возвращал детализацию, 
        # или пока просто total_score.
        # В текущей версии engine возвращает список float (total_score).
        # Для чистоты архитектуры, давай предположим, что engine возвращает кортежи (total, text, meta)
        # или мы посчитаем их здесь? Лучше пусть engine считает всё.
        
        # ВНИМАНИЕ: Так как в прошлом шаге я давал код engine, который возвращал List[float],
        # здесь я адаптируюсь под него, но логичнее расширить engine. 
        # Давай я обновлю вызов, предполагая, что engine теперь умный.
        
        # Для совместимости с текущим engine (v1.0), который возвращает просто float:
        total_scores = scorer_engine.predict_batch(vacancy_full_text, candidates_dicts)
        
        results = []
        for cand_obj, t_score in zip(payload.candidates, total_scores):
            
            # В будущем engine должен возвращать dict {"total": 0.8, "text": 0.7, "meta": 0.9}
            # Пока реконструируем или ставим заглушку для компонентов
            
            results.append(ScoredCandidateResult(
                candidate_id=cand_obj.id,
                total_score=t_score,
                # Пока ставим total, так как движок в прошлой версии не возвращал детализацию.
                # Это можно улучшить в v1.1
                text_score=t_score, 
                meta_score=t_score 
            ))
            
            # Логируем ключевые метрики
            logger.info(f"[{req_id}] CandID: {cand_obj.id} -> Score: {t_score}")

    except Exception as e:
        logger.error(f"[{req_id}] Processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal ML error: {str(e)}")

    logger.info(f"[{req_id}] Batch finished successfully.")

    return ScoringResponse(
        vacancy_title=vacancy_title,
        processed_count=count,
        results=results
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
