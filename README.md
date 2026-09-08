# ML Baseline Reproduction & Experiment Tracking

Фреймворк для воспроизведения ML-экспериментов и сравнения моделей на задаче предсказания оттока клиентов (Customer Churn Prediction). Демонстрирует культуру экспериментов и системный подход к выбору моделей.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8%2B-purple)

## 🎯 Цель эксперимента

**Задача:** Предсказать отток клиентов телеком-компании на основе их поведения и демографических данных.

**Бизнес-контекст:** 
- Стоимость привлечения нового клиента в 5-7 раз выше удержания существующего
- Раннее выявление "уходящих" клиентов позволяет снизить churn rate на 15-25%
- Цель: построить модель с recall > 0.75 для сегмента "at-risk" клиентов

**Гипотеза:** Ансамблевые методы (Random Forest, Gradient Boosting) покажут лучшие результаты по сравнению с линейными моделями благодаря нелинейным зависимостям в данных.

## 📊 Описание данных

**Датасет:** Telco Customer Churn (IBM Sample Data)
- **Размер:** 7,043 записей, 21 признак
- **Целевая переменная:** `Churn` (Yes/No) - бинарная классификация
- **Баланс классов:** 73.5% No / 26.5% Yes (умеренный дисбаланс)

### Признаки:
- **Демография:** gender, SeniorCitizen, Partner, Dependents
- **Услуги:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Аккаунт:** Contract, PaperlessBilling, PaymentMethod
- **Финансы:** MonthlyCharges, TotalCharges, tenure

### Предобработка:
- Кодирование категориальных признаков (One-Hot Encoding)
- Масштабирование числовых признаков (StandardScaler)
- Обработка пропусков в TotalCharges (медиана)

## 🏗️ Архитектура эксперимента

### Модели для сравнения:

1. **Logistic Regression** - базовая линейная модель (baseline)
2. **Random Forest** - ансамбль деревьев, устойчив к переобучению
3. **Gradient Boosting (XGBoost)** - последовательное обучение, высокая точность

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
