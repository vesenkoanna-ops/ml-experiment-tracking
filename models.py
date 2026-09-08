from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import logging

logger = logging.getLogger(__name__)


class ModelFactory:

    @staticmethod
    def get_logistic_regression(class_weight='balanced', random_state=42):
        return LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs'
        )
    
    @staticmethod
    def get_random_forest(class_weight='balanced', random_state=42):
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1
        )
    
    @staticmethod
    def get_gradient_boosting(random_state=42):
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=random_state
        )
    
    @staticmethod
    def get_all_models():
        return {
            'Logistic Regression': ModelFactory.get_logistic_regression(),
            'Random Forest': ModelFactory.get_random_forest(),
            'Gradient Boosting': ModelFactory.get_gradient_boosting()
        }
