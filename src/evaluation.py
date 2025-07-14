import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import os
import sys
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils import load_config

# Setup logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X, y, name, safe_results = False):
    """
    Evaluate a classifier and save metrics and plots.
    """
    try:
        config = load_config()
        output_dir = config['outputs']['figures']

        logger.info(f"Evaluating model: {name}")
        y_pred = model.predict(X)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        cm_counts = confusion_matrix(y, y_pred)
        total = cm_counts.sum()
        cm_percent = (cm_counts / total * 100).round(2)
        annot = [[f'{cm[i,j]}\n({cm_percent[i,j]:.2f}%)' for j in range(cm.shape[1])] for i in range(cm.shape[0])]

        # Create plots
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix\n{name}')

        # ROC Curve
        roc_auc = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.subplot(1, 2, 2)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve\n{name}')
            plt.legend(loc="lower right")
        else:
            plt.text(0.5, 0.5, 'ROC curve not available\nfor this model', ha='center', va='center', fontsize=12)
            plt.title(f'ROC Curve\n{name}')

        if safe_results:
            # Save plot
            os.makedirs(output_dir, exist_ok = True)
            plot_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}_plot.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Plot saved to {plot_path}")

        # Log metrics
        logger.info(f"\n{name} Results:")
        logger.info("=" * 50)
        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        if roc_auc is not None:
            logger.info(f"ROC-AUC:   {roc_auc:.4f}")

        if safe_results:
            # Save metrics to file
            metrics_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}_metrics.txt")
            with open(metrics_path, 'w') as f:
                f.write(f"{name} Results:\n")
                f.write("=" * 50 + "\n")
                f.write(f"Accuracy:  {accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall:    {recall:.4f}\n")
                f.write(f"F1-Score:  {f1:.4f}\n")
                if roc_auc is not None:
                    f.write(f"ROC-AUC:   {roc_auc:.4f}\n")
                f.write("\nClassification Report:\n")
                f.write(classification_report(y, y_pred))
            logger.info(f"Metrics saved to {metrics_path}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise