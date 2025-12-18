import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from typing import List, Dict, Any

# Импортируем схемы для аннотации типов (если нужно)
# from schemas import ScoringRequest 

class TrainableLRScorer:
    def __init__(self):
        self.pipeline = None
        self.is_trained = False

    def _concat_features(self, vacancy_title: str, vacancy_desc: str, candidate_dict: Dict[str, Any]) -> str:
        """
        Собираем все поля в одну длинную строку.
        Формат: "VACANCY: title desc CANDIDATE: field1 field2 ..."
        """
        # Собираем данные кандидата в строку, пропуская пустые поля
        cand_text_parts = []
        
        # Список полей, которые мы хотим включить в обучение
        relevant_fields = [
            "cv_text", "stack_keywords", "university", 
            "city", "schedule", "education_level", "major"
        ]
        
        for key in relevant_fields:
            val = candidate_dict.get(key)
            if val:
                cand_text_parts.append(str(val))
        
        cand_full_text = " ".join(cand_text_parts)
        
        # Формируем итоговый документ для TF-IDF
        # Добавляем префиксы, чтобы модель могла (теоретически) различать секции, 
        # хотя для bag-of-words это не критично.
        full_text = f"VACANCY_TITLE: {vacancy_title} VACANCY_DESC: {vacancy_desc} CANDIDATE_DATA: {cand_full_text}"
        return full_text.lower().strip()

    def generate_dummy_data(self):
        """
        Генерация 5 обучающих примеров.
        Возвращает X (тексты) и y (целевые скоры).
        """
        samples = [
            # 1. Идеальное совпадение (Python -> Python)
            {
                "vac_title": "Python Developer",
                "vac_desc": "Django FastAPI Docker",
                "cand": {"cv_text": "Experienced Python Backend Developer Django FastAPI", "stack_keywords": "Python; SQL"},
                "target": 0.95
            },
            # 2. Хорошее совпадение (Data Analyst -> Analyst)
            {
                "vac_title": "Data Analyst",
                "vac_desc": "SQL Python Pandas",
                "cand": {"cv_text": "Data Analyst with SQL and Excel skills", "stack_keywords": "SQL; Excel"},
                "target": 0.85
            },
            # 3. Частичное совпадение (Python -> Data Scientist)
            {
                "vac_title": "Python Developer",
                "vac_desc": "Backend services",
                "cand": {"cv_text": "Data Scientist trained models", "stack_keywords": "Python; PyTorch"},
                "target": 0.45
            },
            # 4. Плохое совпадение (Java -> Python)
            {
                "vac_title": "Java Developer",
                "vac_desc": "Spring Boot Hibernate",
                "cand": {"cv_text": "Python developer Django", "stack_keywords": "Python"},
                "target": 0.10
            },
            # 5. Мусор / Несовпадение (Driver -> Manager)
            {
                "vac_title": "Personal Driver",
                "vac_desc": "Driving license B category",
                "cand": {"cv_text": "Sales Manager selling software", "stack_keywords": "Sales"},
                "target": 0.01
            }
        ]

        X = []
        y = []
        
        for s in samples:
            text = self._concat_features(s["vac_title"], s["vac_desc"], s["cand"])
            X.append(text)
            y.append(s["target"])
            
        return X, y

    def train(self):
        """Процесс обучения пайплайна"""
        X_train, y_train = self.generate_dummy_data()
        
        # Создаем пайплайн: TF-IDF -> Линейная регрессия
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('regressor', LinearRegression())
        ])
        
        print(f"Training model on {len(X_train)} samples...")
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        print("Model training completed.")
        
        # Выведем тест на обучающей выборке для отладки
        test_preds = self.pipeline.predict(X_train)
        print(f"Training Check (Targets vs Preds): {list(zip(y_train, np.round(test_preds, 2)))}")

    def predict(self, vacancy_title: str, vacancy_desc: str, candidates_dicts: List[Dict]) -> List[float]:
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")

        # Подготовка данных
        X_pred = []
        for cand in candidates_dicts:
            text = self._concat_features(vacancy_title, vacancy_desc or "", cand)
            X_pred.append(text)
        
        # Инференс
        scores = self.pipeline.predict(X_pred)
        
        # LinearRegression может вернуть < 0 или > 1, нужно обрезать (clip)
        scores_clipped = np.clip(scores, 0.0, 1.0)
        
        return scores_clipped.tolist()

# Создаем инстанс (синглтон)
lr_scorer = TrainableLRScorer()
