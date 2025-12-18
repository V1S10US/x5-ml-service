from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Any

# ---------------------------------------------------------
# 1. Vacancy Models
# ---------------------------------------------------------
class VacancyData(BaseModel):
    title: str = Field(..., description="Название вакансии (Target)")
    description: Optional[str] = Field("", description="Полное описание вакансии")

# ---------------------------------------------------------
# 2. Candidate Models
# ---------------------------------------------------------
class CandidateFeatures(BaseModel):
    """
    Модель кандидата с нормализованными английскими именами полей.
    Маппинг из Excel (рус) -> API (eng) должен происходить НА КЛИЕНТЕ 
    или в сервисе-оркестраторе перед отправкой сюда.
    """
    # Обязательные идентификаторы
    id: str = Field(..., description="Уникальный ID кандидата")
    
    # Ключевой контент для NLP
    cv_text: str = Field("", description="Полный текст резюме после OCR")
    
    # Метаданные для эвристик (Meta Score)
    # Маппинг: 'Языки' -> stack_keywords
    stack_keywords: Optional[str] = Field(None, description="Стек технологий из анкеты")
    
    # Маппинг: 'ВУЗ' -> university
    university: Optional[str] = Field(None, description="Название ВУЗа")
    
    # Маппинг: 'Город' -> city
    city: Optional[str] = Field(None, description="Город проживания")
    
    # Маппинг: 'График' -> schedule
    schedule: Optional[str] = Field(None, description="Желаемый график (40 часов и т.д.)")
    
    # Маппинг: 'Курс' -> education_level
    education_level: Optional[str] = Field(None, description="Курс или год выпуска")

    # Маппинг: 'Специальность' -> major
    major: Optional[str] = Field(None, description="Специальность по диплому")

    # Дополнительные поля (Фамилия, Имя, ТГ) - не влияют на скор напрямую, но полезны для логов
    first_name: Optional[str] = Field(None, alias="first_name")
    last_name: Optional[str] = Field(None, alias="last_name")
    
    # Разрешаем любые другие поля (citizenship, birth_year и т.д.)
    model_config = ConfigDict(extra='allow')

# ---------------------------------------------------------
# 3. Request/Response Models
# ---------------------------------------------------------
class ScoringRequest(BaseModel):
    vacancy: VacancyData
    candidates: List[CandidateFeatures]

class ScoredCandidateResult(BaseModel):
    candidate_id: str
    total_score: float = Field(..., description="Итоговая релевантность (0-1)")
    # Можно возвращать компоненты скора для прозрачности
    text_score: float
    meta_score: float

class ScoringResponse(BaseModel):
    vacancy_title: str
    processed_count: int
    results: List[ScoredCandidateResult]


# --- Расширенные модели ответов для LLM ---
class ScoredCandidateResultLLM(ScoredCandidateResult):
    reason: str = Field("", description="Текстовое обоснование оценки от LLM")

class ScoringResponseLLM(BaseModel):
    vacancy_title: str
    processed_count: int
    results: List[ScoredCandidateResultLLM]