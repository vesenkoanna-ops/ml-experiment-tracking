import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
from pathlib import Path

from data_loader import ChurnDataLoader
from models import ModelFactory

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
 
    def __init__(self, data_source='openml'):
        self.data_loader = ChurnDataLoader(data_source)
        self.models = ModelFactory.get_all_models()
        self.results = []
        
    def run_experiment(self):
        logger.info("=" * 60)
        logger.info("ЗАПУСК ЭКСПЕРИМЕНТА: Customer Churn Prediction")
        logger.info("=" * 60)

        self.data_loader.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.preprocess()

        for model_name, model in self.models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Модель: {model_name}")
            logger.info(f"{'='*60}")

            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
                'Training_Time': training_time
            }
            
            self.results.append(metrics)

            logger.info(f"Accuracy:  {metrics['Accuracy']:.4f}")
            logger.info(f"Precision: {metrics['Precision']:.4f}")
            logger.info(f"Recall:    {metrics['Recall']:.4f}")
            logger.info(f"F1-Score:  {metrics['F1-Score']:.4f}")
            logger.info(f"ROC-AUC:   {metrics['ROC-AUC']:.4f}")
            logger.info(f"Время обучения: {training_time:.2f}s")

            logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"Confusion Matrix:\n{cm}")

        results_df = pd.DataFrame(self.results)
        results_df.to_csv('results/metrics.csv', index=False)
        logger.info(f"\nРезультаты сохранены в results/metrics.csv")

        self._plot_metrics(results_df)

        best_model = results_df.loc[results_df['F1-Score'].idxmax()]
        logger.info(f"\n{'='*60}")
        logger.info(f"ЛУЧШАЯ МОДЕЛЬ: {best_model['Model']}")
        logger.info(f"F1-Score: {best_model['F1-Score']:.4f}")
        logger.info(f"{'='*60}")
        
        return results_df
    
    def _plot_metrics(self, results_df):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Comparison: Customer Churn Prediction', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#2196F3', '#4CAF50', '#FF9800']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            bars = ax.bar(results_df['Model'], results_df[metric], color=colors, edgecolor='black', alpha=0.8)
            ax.set_title(metric, fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)

            for bar, value in zip(bars, results_df[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/metrics_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("График сохранен: results/metrics_comparison.png")
        plt.show()

        best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
        if best_model_name != 'Logistic Regression':
            self._plot_feature_importance(best_model_name)
    
    def _plot_feature_importance(self, model_name):
        """Визуализация важности признаков"""
        model = self.models[model_name]
        feature_names = self.data_loader.get_feature_names()
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10
            
            plt.figure(figsize=(12, 6))
            plt.title(f'Top 10 Feature Importances - {model_name}', fontsize=14, fontweight='bold')
            plt.barh(range(len(indices)), importances[indices], color='#2196F3', edgecolor='black')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
            logger.info("Feature importance сохранен: results/feature_importance.png")
            plt.show()


def main():
    Path('results').mkdir(exist_ok=True)

    runner = ExperimentRunner(data_source='openml')
    results_df = runner.run_experiment()
    
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЕН УСПЕШНО")
    print("="*60)
    print(f"\n📊 Результаты:")
    print(results_df.to_string(index=False))
    print(f"\n📁 Файлы результатов:")
    print(f"   - results/metrics.csv")
    print(f"   - results/metrics_comparison.png")
    print(f"   - results/feature_importance.png")


if __name__ == "__main__":
    main()
