from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any

class VacancyNode(BaseModel):
    title: str = Field(..., alias="название вакансии", description="Название позиции")
    description: str = Field(default="", alias="Описание вакансии", description="Полный текст вакансии")

    model_config = ConfigDict(populate_by_name=True)

class CandidateNode(BaseModel):
    id: str = Field(..., description="Уникальный ID кандидата (строка)")
    cv_text: str = Field(default="", description="Распаршенный текст резюме")
    
    # Позволяет принимать любые дополнительные поля из Excel (ВУЗ, График и т.д.)
    # без жесткого описания в схеме
    model_config = ConfigDict(extra='allow')

class ScoringRequest(BaseModel):
    vacancy: VacancyNode = Field(..., alias="вакансия")
    candidates: List[CandidateNode] = Field(..., alias="кандидаты")

    model_config = ConfigDict(populate_by_name=True)

class ScoredCandidate(BaseModel):
    candidate_id: str
    score: float
    status: str = "processed"

class ScoringResponse(BaseModel):
    vacancy_title: str
    results: List[ScoredCandidate]

