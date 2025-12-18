import os
import json
import httpx
from typing import List, Dict, Any
from custom_logger import logger
from dotenv import load_dotenv

load_dotenv()
class LLMScorer:
    def __init__(self):
        # Настройки API (задаются через ENV или хардкод для теста)
        self.api_key = os.getenv("LLM_API_KEY", "sk-xxx-placeholder")
        self.api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
        self.model_name = os.getenv("LLM_MODEL", "gpt-3.5-turbo") # или gpt-4o / vllm local model

    def _construct_prompt(self, vacancy_title: str, vacancy_desc: str, candidate: Dict[str, Any]) -> str:
        """
        Собираем один большой промпт для модели.
        """
        # Формируем читаемое представление кандидата
        cand_details = "\n".join([f"- {k}: {v}" for k, v in candidate.items() if v and k != "cv_text"])
        cv_excerpt = (candidate.get("cv_text") or "")[:2000] # Обрезаем, чтобы не переполнить контекст

        prompt = f"""
Ты профессиональный IT-рекрутер. Твоя задача — оценить кандидата на вакансию по шкале от 0.0 до 1.0.

ВАКАНСИЯ:
Название: {vacancy_title}
Описание: {vacancy_desc}

КАНДИДАТ:
{cand_details}
Текст резюме (фрагмент):
{cv_excerpt}

ИНСТРУКЦИЯ:
1. Проанализируй стек технологий, опыт и образование.
2. Сравни с требованиями вакансии.
3. Верни JSON с одним полем "score" (число float от 0.0 до 1.0) и кратким обоснованием "reason".
4. Если кандидат совсем не подходит, ставь < 0.3. Если идеален, > 0.8.

ФОРМАТ ОТВЕТА (JSON ONLY):
{{
  "score": 0.75,
  "reason": "Есть релевантный опыт Python, но не знает Docker."
}}
"""
        return prompt

    async def predict_one(self, vacancy_title: str, vacancy_desc: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Асинхронный вызов API для одного кандидата.
        """
        prompt = self._construct_prompt(vacancy_title, vacancy_desc, candidate)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful HR assistant returning strict JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Внимание: для реального вызова нужен валидный ключ
                # Если ключа нет, ставим заглушку для теста, чтобы код не падал
                if self.api_key == "sk-xxx-placeholder":
                    # --- MOCK RESPONSE FOR TESTING WITHOUT KEY ---
                    import asyncio
                    await asyncio.sleep(0.5) # Имитация задержки сети
                    return {"score": 0.88, "reason": "MOCKED LLM RESPONSE: Good match based on keywords."}
                    # ---------------------------------------------

                resp = await client.post(f"{self.api_base}/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                
                content = data["choices"][0]["message"]["content"]
                
                # Пытаемся распарсить JSON из ответа модели
                # Иногда модели добавляют Markdown ``````, чистим это
                clean_content = content.replace("``````", "").strip()
                return json.loads(clean_content)

        except Exception as e:
            logger.error(f"LLM Call failed for candidate {candidate.get('id')}: {e}")
            return {"score": 0.0, "reason": f"Error: {str(e)}"}

# Синглтон
llm_scorer = LLMScorer()
