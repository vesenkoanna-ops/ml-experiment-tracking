import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


class ChurnDataLoader:
    
    def __init__(self, data_source='openml'):
        self.data_source = data_source
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def load_data(self, csv_path=None):
        logger.info("Загрузка данных...")
        
        if self.data_source == 'openml':
            # Загрузка с OpenML (Telco Customer Churn)
            from sklearn.datasets import fetch_openml
            try:
                data = fetch_openml(data_id=42174, as_frame=True, parser='auto')
                self.df = data.frame
                logger.info(f"Загружено {len(self.df)} записей с OpenML")
            except Exception as e:
                logger.warning(f"Не удалось загрузить с OpenML: {e}")
                logger.info("Использую синтетические данные для демонстрации")
                self.df = self._generate_synthetic_data()
        elif self.data_source == 'csv' and csv_path:
            self.df = pd.read_csv(csv_path)
            logger.info(f"Загружено {len(self.df)} записей из {csv_path}")
        else:
            raise ValueError("Укажите data_source='openml' или data_source='csv' с csv_path")
        
        return self.df
    
    def _generate_synthetic_data(self, n_samples=7043):
        np.random.seed(42)
        
        data = {
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.48, 0.52]),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70]),
            'tenure': np.random.randint(1, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.42, 0.48, 0.10]),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22]),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.49, 0.21]),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.49, 0.21]),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.49, 0.21]),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.49, 0.21]),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.49, 0.21]),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.49, 0.21]),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.21, 0.24]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.60, 0.40]),
            'PaymentMethod': np.random.choice(
                ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                n_samples, p=[0.34, 0.23, 0.22, 0.21]
            ),
            'MonthlyCharges': np.random.uniform(18.25, 118.75, n_samples).round(2),
            'TotalCharges': np.random.uniform(18.80, 8684.80, n_samples).round(2),
        }
        
        churn_prob = np.zeros(n_samples)
        churn_prob += (data['tenure'] < 12) * 0.3
        churn_prob += (data['MonthlyCharges'] > 70) * 0.2
        churn_prob += (data['Contract'] == 'Month-to-month') * 0.25
        churn_prob += (data['InternetService'] == 'Fiber optic') * 0.15
        churn_prob += (data['OnlineSecurity'] == 'No') * 0.1
        churn_prob += (data['TechSupport'] == 'No') * 0.1

        churn_prob = np.clip(churn_prob / churn_prob.max(), 0, 1)
        data['Churn'] = np.random.binomial(1, churn_prob * 0.4)  # ~26% churn rate
        
        self.df = pd.DataFrame(data)
        return self.df
    
    def preprocess(self, test_size=0.15, val_size=0.15, random_state=42):
        logger.info("Начало предобработки...")
        
        df = self.df.copy()

        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

        le = LabelEncoder()
        df['Churn'] = le.fit_transform(df['Churn'])  # 0 = No, 1 = Yes

        X = df.drop('Churn', axis=1)
        y = df['Churn']

        categorical_cols = X.select_dtypes(include=['object']).columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )

        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.scaler.fit(self.X_train[numeric_cols])
        
        self.X_train[numeric_cols] = self.scaler.transform(self.X_train[numeric_cols])
        self.X_val[numeric_cols] = self.scaler.transform(self.X_val[numeric_cols])
        self.X_test[numeric_cols] = self.scaler.transform(self.X_test[numeric_cols])
        
        logger.info(f"Размер данных:")
        logger.info(f"  Train: {len(self.X_train)} записей")
        logger.info(f"  Validation: {len(self.X_val)} записей")
        logger.info(f"  Test: {len(self.X_test)} записей")
        logger.info(f"  Признаков: {self.X_train.shape[1]}")
        logger.info(f"  Баланс классов (train): {self.y_train.value_counts().to_dict()}")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def get_feature_names(self):
        return self.X_train.columns.tolist()
