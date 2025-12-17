import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import re

class SimpleScorer:
    def __init__(self):
        # В реальном проде мы бы загружали предобученный vectorizer
        # Но для бейзлайна обучим его "на лету" на батче + вакансии
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def _preprocess(self, text: str) -> str:
        """Простая очистка текста"""
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zа-я0-9\s]', '', text) # Убираем спецсимволы
        return text

    def compute_text_similarity(self, vacancy_text: str, resumes: List[str]) -> List[float]:
        """Считает TF-IDF Cosine Similarity между вакансией и резюме"""
        if not vacancy_text:
            # Если текста вакансии нет совсем, возвращаем нейтральный скор
            return [0.5] * len(resumes)
        
        # Собираем корпус: [Вакансия, Резюме 1, Резюме 2, ...]
        corpus = [self._preprocess(vacancy_text)] + [self._preprocess(r) for r in resumes]
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            # Считаем косинусное расстояние первой строки (вакансии) ко всем остальным
            # tfidf_matrix[0:1] - это вектор вакансии
            # tfidf_matrix[1:] - это матрица векторов резюме
            cosine_sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            return cosine_sims.tolist()
        except ValueError:
            # Если корпус пустой или состоит только из стоп-слов
            return [0.0] * len(resumes)

    def compute_meta_score(self, candidate_data: Dict) -> float:
        """Эвристическая оценка метаданных (0.0 - 1.0)"""
        score = 0.5 # Базовый скор
        
        # 1. Проверка ВУЗа (Пример списка топовых вузов)
        top_universities = ["мгу", "вшэ", "мфти", "спбгу", "итмо", "бауман"]
        uni = str(candidate_data.get("university", "")).lower()
        if any(u in uni for u in top_universities):
            score += 0.2
            
        # 2. Проверка города (Бонус за Москву/Питер, если не указано иное)
        # В идеале нужно сравнивать с городом вакансии, но пока хардкод
        city = str(candidate_data.get("city", "")).lower()
        if "москва" in city or "санкт-петербург" in city:
            score += 0.1
            
        # 3. График работы (Бонус за фултайм)
        schedule = str(candidate_data.get("work_hours", "")).lower()
        if "40" in schedule or "полн" in schedule:
            score += 0.1
            
        return min(1.0, score) # Cap at 1.0

    def predict_batch(self, vacancy_full_text: str, candidates: List[Dict]) -> List[float]:
        """Основной метод пайплайна"""
        
        # 1. Вытаскиваем тексты резюме
        resume_texts = [c.get("cv_text", "") for c in candidates]
        
        # 2. Считаем текстовую похожесть
        text_scores = self.compute_text_similarity(vacancy_full_text, resume_texts)
        
        final_scores = []
        for i, cand_dict in enumerate(candidates):
            # 3. Считаем мета-скор для каждого
            meta_score = self.compute_meta_score(cand_dict)
            
            # 4. Ансамблирование (Weighted Sum)
            # Если текста резюме мало, доверяем больше метаданным
            if len(resume_texts[i]) < 50:
                 final_score = 0.2 * text_scores[i] + 0.8 * meta_score
            else:
                 final_score = 0.7 * text_scores[i] + 0.3 * meta_score
            
            final_scores.append(round(final_score, 3))
            
        return final_scores

# Синглтон для импорта
scorer_engine = SimpleScorer()

