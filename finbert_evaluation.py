import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, cross_val_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Dict, Tuple, Union, Optional
import json
from tqdm import tqdm
import scipy.stats as stats

class FinBERTEvaluator:
    """
    A comprehensive evaluator for assessing FinBERT model performance
    with statistical precision measures.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = None,
        labels: List[str] = None
    ):
        """
        Initialize the FinBERT evaluator with a specified model.

        Args:
            model_name: HuggingFace model identifier for the finBERT model
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detection)
            labels: Custom label list if different from default ['negative', 'neutral', 'positive']
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Default finBERT labels
        self.labels = labels if labels else ['negative', 'neutral', 'positive']

        # Initialize metrics storage
        self.results = {}
        self.bootstrap_results = {}
        self.cross_val_results = {}

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Generate predictions for a list of financial texts.

        Args:
            texts: List of financial text samples to classify

        Returns:
            Numpy array of predicted class indices
        """
        predictions = []

        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(batch_predictions)

        return np.array(predictions)

    def evaluate(
        self,
        texts: List[str],
        true_labels: List[Union[int, str]],
        detailed: bool = True
    ) -> Dict:
        """
        Evaluate model performance on test data.

        Args:
            texts: List of financial text samples
            true_labels: Ground truth labels (can be indices or label strings)
            detailed: Whether to compute detailed metrics

        Returns:
            Dictionary of evaluation metrics
        """
        # Convert string labels to indices if needed
        if isinstance(true_labels[0], str):
            label_map = {label: i for i, label in enumerate(self.labels)}
            true_labels = [label_map.get(label, -1) for label in true_labels]

        # Ensure all labels are valid
        if any(l < 0 or l >= len(self.labels) for l in true_labels):
            raise ValueError("Some true labels are not in the model's label set")

        # Get predictions
        predictions = self.predict(texts)

        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels,
            predictions,
            average=None,
            labels=range(len(self.labels))
        )

        # Weighted metrics
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            true_labels,
            predictions,
            average='weighted'
        )

        # Prepare results
        self.results = {
            'accuracy': accuracy,
            'class_metrics': {
                label: {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i],
                    'support': support[i]
                } for i, label in enumerate(self.labels)
            },
            'weighted_metrics': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1': weighted_f1
            }
        }

        if detailed:
            # Confusion matrix
            cm = confusion_matrix(true_labels, predictions)
            self.results['confusion_matrix'] = cm.tolist()

            # Per-sample predictions
            self.results['sample_results'] = [{
                'text': text[:100] + ('...' if len(text) > 100 else ''),
                'true_label': self.labels[true_label],
                'predicted_label': self.labels[pred],
                'correct': true_label == pred
            } for text, true_label, pred in zip(texts, true_labels, predictions)]

        return self.results

    def bootstrap_confidence_intervals(
        self,
        texts: List[str],
        true_labels: List[Union[int, str]],
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 42
    ) -> Dict:
        """
        Calculate confidence intervals for evaluation metrics using bootstrapping.

        Args:
            texts: List of financial text samples
            true_labels: Ground truth labels
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level for intervals (0-1)
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with confidence intervals for each metric
        """
        np.random.seed(random_seed)

        # Convert string labels to indices if needed
        if isinstance(true_labels[0], str):
            label_map = {label: i for i, label in enumerate(self.labels)}
            true_labels = [label_map.get(label, -1) for label in true_labels]

        # Initial evaluation to get all predictions at once
        predictions = self.predict(texts)

        # Prepare arrays for storing bootstrap results
        bootstrap_accuracies = np.zeros(n_iterations)
        bootstrap_precision = np.zeros((n_iterations, len(self.labels)))
        bootstrap_recall = np.zeros((n_iterations, len(self.labels)))
        bootstrap_f1 = np.zeros((n_iterations, len(self.labels)))
        bootstrap_weighted_precision = np.zeros(n_iterations)
        bootstrap_weighted_recall = np.zeros(n_iterations)
        bootstrap_weighted_f1 = np.zeros(n_iterations)

        # Bootstrapping
        sample_size = len(texts)
        for i in tqdm(range(n_iterations), desc="Bootstrapping"):
            # Generate bootstrap sample indices
            indices = np.random.choice(sample_size, sample_size, replace=True)

            # Get bootstrap samples
            bootstrap_true = np.array(true_labels)[indices]
            bootstrap_pred = predictions[indices]

            # Compute metrics for this bootstrap sample
            bootstrap_accuracies[i] = accuracy_score(bootstrap_true, bootstrap_pred)

            # Class-specific metrics
            p, r, f, _ = precision_recall_fscore_support(
                bootstrap_true,
                bootstrap_pred,
                average=None,
                labels=range(len(self.labels)),
                zero_division=0
            )
            bootstrap_precision[i] = p
            bootstrap_recall[i] = r
            bootstrap_f1[i] = f

            # Weighted metrics
            wp, wr, wf, _ = precision_recall_fscore_support(
                bootstrap_true,
                bootstrap_pred,
                average='weighted',
                zero_division=0
            )
            bootstrap_weighted_precision[i] = wp
            bootstrap_weighted_recall[i] = wr
            bootstrap_weighted_f1[i] = wf

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        # Prepare results
        self.bootstrap_results = {
            'parameters': {
                'n_iterations': n_iterations,
                'confidence_level': confidence_level,
                'sample_size': sample_size
            },
            'accuracy': {
                'mean': np.mean(bootstrap_accuracies),
                'std': np.std(bootstrap_accuracies),
                'ci_lower': np.percentile(bootstrap_accuracies, lower_percentile),
                'ci_upper': np.percentile(bootstrap_accuracies, upper_percentile)
            },
            'class_metrics': {
                label: {
                    'precision': {
                        'mean': np.mean(bootstrap_precision[:, i]),
                        'std': np.std(bootstrap_precision[:, i]),
                        'ci_lower': np.percentile(bootstrap_precision[:, i], lower_percentile),
                        'ci_upper': np.percentile(bootstrap_precision[:, i], upper_percentile)
                    },
                    'recall': {
                        'mean': np.mean(bootstrap_recall[:, i]),
                        'std': np.std(bootstrap_recall[:, i]),
                        'ci_lower': np.percentile(bootstrap_recall[:, i], lower_percentile),
                        'ci_upper': np.percentile(bootstrap_recall[:, i], upper_percentile)
                    },
                    'f1': {
                        'mean': np.mean(bootstrap_f1[:, i]),
                        'std': np.std(bootstrap_f1[:, i]),
                        'ci_lower': np.percentile(bootstrap_f1[:, i], lower_percentile),
                        'ci_upper': np.percentile(bootstrap_f1[:, i], upper_percentile)
                    }
                } for i, label in enumerate(self.labels)
            },
            'weighted_metrics': {
                'precision': {
                    'mean': np.mean(bootstrap_weighted_precision),
                    'std': np.std(bootstrap_weighted_precision),
                    'ci_lower': np.percentile(bootstrap_weighted_precision, lower_percentile),
                    'ci_upper': np.percentile(bootstrap_weighted_precision, upper_percentile)
                },
                'recall': {
                    'mean': np.mean(bootstrap_weighted_recall),
                    'std': np.std(bootstrap_weighted_recall),
                    'ci_lower': np.percentile(bootstrap_weighted_recall, lower_percentile),
                    'ci_upper': np.percentile(bootstrap_weighted_recall, upper_percentile)
                },
                'f1': {
                    'mean': np.mean(bootstrap_weighted_f1),
                    'std': np.std(bootstrap_weighted_f1),
                    'ci_lower': np.percentile(bootstrap_weighted_f1, lower_percentile),
                    'ci_upper': np.percentile(bootstrap_weighted_f1, upper_percentile)
                }
            }
        }

        return self.bootstrap_results

    def cross_validate(
        self,
        texts: List[str],
        true_labels: List[Union[int, str]],
        n_splits: int = 5,
        random_seed: int = 42
    ) -> Dict:
        """
        Perform cross-validation evaluation.

        Args:
            texts: List of financial text samples
            true_labels: Ground truth labels
            n_splits: Number of cross-validation folds
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with cross-validation results
        """
        # Convert string labels to indices if needed
        if isinstance(true_labels[0], str):
            label_map = {label: i for i, label in enumerate(self.labels)}
            true_labels = [label_map.get(label, -1) for label in true_labels]

        # Convert to numpy arrays
        texts_array = np.array(texts)
        labels_array = np.array(true_labels)

        # Initialize cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

        # Metrics for each fold
        fold_results = []

        # For computing average metrics across folds
        all_true_labels = []
        all_predictions = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(texts_array)):
            # Get test split for this fold
            fold_texts = texts_array[test_idx]
            fold_true = labels_array[test_idx]

            # Get predictions
            fold_preds = self.predict(fold_texts.tolist())

            # Store for overall metrics
            all_true_labels.extend(fold_true)
            all_predictions.extend(fold_preds)

            # Calculate fold metrics
            fold_accuracy = accuracy_score(fold_true, fold_preds)
            fold_precision, fold_recall, fold_f1, _ = precision_recall_fscore_support(
                fold_true,
                fold_preds,
                average='weighted'
            )

            # Store fold results
            fold_results.append({
                'fold': fold + 1,
                'accuracy': fold_accuracy,
                'precision': fold_precision,
                'recall': fold_recall,
                'f1': fold_f1
            })

        # Calculate overall metrics
        overall_accuracy = accuracy_score(all_true_labels, all_predictions)
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            all_true_labels,
            all_predictions,
            average='weighted'
        )

        # Calculate fold statistics
        accuracies = [fold['accuracy'] for fold in fold_results]
        precisions = [fold['precision'] for fold in fold_results]
        recalls = [fold['recall'] for fold in fold_results]
        f1s = [fold['f1'] for fold in fold_results]

        # Store cross-validation results
        self.cross_val_results = {
            'parameters': {
                'n_splits': n_splits
            },
            'fold_results': fold_results,
            'overall_metrics': {
                'accuracy': overall_accuracy,
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1
            },
            'fold_statistics': {
                'accuracy': {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'min': np.min(accuracies),
                    'max': np.max(accuracies)
                },
                'precision': {
                    'mean': np.mean(precisions),
                    'std': np.std(precisions),
                    'min': np.min(precisions),
                    'max': np.max(precisions)
                },
                'recall': {
                    'mean': np.mean(recalls),
                    'std': np.std(recalls),
                    'min': np.min(recalls),
                    'max': np.max(recalls)
                },
                'f1': {
                    'mean': np.mean(f1s),
                    'std': np.std(f1s),
                    'min': np.min(f1s),
                    'max': np.max(f1s)
                }
            }
        }

        return self.cross_val_results

    def error_analysis(
        self,
        texts: List[str],
        true_labels: List[Union[int, str]],
        n_samples: int = 10
    ) -> Dict:
        """
        Perform detailed error analysis to identify patterns in misclassifications.

        Args:
            texts: List of financial text samples
            true_labels: Ground truth labels
            n_samples: Number of misclassification examples to return

        Returns:
            Dictionary with error analysis results
        """
        # Ensure evaluation has been performed
        if not self.results:
            self.evaluate(texts, true_labels)

        # Convert string labels to indices if needed
        if isinstance(true_labels[0], str):
            label_map = {label: i for i, label in enumerate(self.labels)}
            true_labels = [label_map.get(label, -1) for label in true_labels]

        # Get predictions if not already available
        predictions = self.predict(texts)

        # Find misclassified examples
        misclassified_indices = np.where(np.array(true_labels) != predictions)[0]
        misclassified_count = len(misclassified_indices)

        # Calculate error rate by class
        error_rate_by_class = {}
        for i, label in enumerate(self.labels):
            class_indices = np.where(np.array(true_labels) == i)[0]
            if len(class_indices) > 0:
                class_errors = np.sum(predictions[class_indices] != i)
                error_rate = class_errors / len(class_indices)
                error_rate_by_class[label] = error_rate

        # Calculate confusion pairs (actual → predicted)
        confusion_pairs = {}
        for true_label in range(len(self.labels)):
            true_indices = np.where(np.array(true_labels) == true_label)[0]
            for pred_label in range(len(self.labels)):
                if true_label != pred_label:
                    count = np.sum((predictions[true_indices] == pred_label))
                    if count > 0:
                        pair = f"{self.labels[true_label]} → {self.labels[pred_label]}"
                        confusion_pairs[pair] = {
                            'count': int(count),
                            'percentage': float(count / len(true_indices) * 100)
                        }

        # Get sample misclassifications
        sample_misclassifications = []
        if misclassified_count > 0:
            # Randomly sample misclassifications
            sample_indices = np.random.choice(
                misclassified_indices,
                min(n_samples, misclassified_count),
                replace=False
            )

            for idx in sample_indices:
                sample_misclassifications.append({
                    'text': texts[idx],
                    'true_label': self.labels[true_labels[idx]],
                    'predicted_label': self.labels[predictions[idx]]
                })

        # Return error analysis results
        error_analysis_results = {
            'total_samples': len(texts),
            'misclassified_count': misclassified_count,
            'error_rate': misclassified_count / len(texts),
            'error_rate_by_class': error_rate_by_class,
            'confusion_pairs': confusion_pairs,
            'sample_misclassifications': sample_misclassifications
        }

        return error_analysis_results

    def benchmark_comparison(
        self,
        benchmark_results: Dict[str, Dict],
        metric: str = 'f1'
    ) -> Dict:
        """
        Compare model performance against benchmarks.

        Args:
            benchmark_results: Dictionary of benchmark results by model name
            metric: Main metric for comparison ('accuracy', 'precision', 'recall', 'f1')

        Returns:
            Dictionary with comparison results
        """
        if not self.results:
            raise ValueError("Must run evaluate() before comparing with benchmarks")

        # Get our model's metric
        if metric == 'accuracy':
            our_metric = self.results['accuracy']
        else:
            our_metric = self.results['weighted_metrics'][metric]

        # Prepare comparison data
        comparisons = {
            'main_metric': metric,
            'our_model': {
                'name': self.model_name,
                metric: our_metric
            },
            'benchmark_models': []
        }

        # Add benchmark data
        for model_name, results in benchmark_results.items():
            if metric == 'accuracy':
                benchmark_metric = results.get('accuracy')
            else:
                benchmark_metric = results.get('weighted_metrics', {}).get(metric)

            if benchmark_metric is not None:
                diff = our_metric - benchmark_metric
                comparisons['benchmark_models'].append({
                    'name': model_name,
                    metric: benchmark_metric,
                    'difference': diff,
                    'better_than_benchmark': diff > 0
                })

        # Sort benchmarks by performance
        comparisons['benchmark_models'].sort(key=lambda x: x[metric], reverse=True)

        # Calculate ranking
        all_models = [{'name': 'our_model', metric: our_metric}] + comparisons['benchmark_models']
        all_models.sort(key=lambda x: x[metric], reverse=True)
        our_rank = next(i+1 for i, model in enumerate(all_models) if model['name'] == 'our_model')
        comparisons['our_rank'] = our_rank
        comparisons['total_models'] = len(all_models)

        return comparisons

    def plot_confusion_matrix(self, normalized: bool = True, save_path: Optional[str] = None):
        """
        Plot the confusion matrix from evaluation results.

        Args:
            normalized: Whether to normalize the confusion matrix
            save_path: Path to save the figure (if None, will show the figure)
        """
        if 'confusion_matrix' not in self.results:
            raise ValueError("Must run evaluate() with detailed=True before plotting confusion matrix")

        cm = np.array(self.results['confusion_matrix'])

        if normalized:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.labels,
            yticklabels=self.labels
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def plot_metrics_comparison(self, save_path: Optional[str] = None):
        """
        Plot comparison of precision, recall, and F1 across classes.

        Args:
            save_path: Path to save the figure (if None, will show the figure)
        """
        if not self.results:
            raise ValueError("Must run evaluate() before plotting metrics")

        # Extract metrics for each class
        classes = []
        precision_values = []
        recall_values = []
        f1_values = []

        for label, metrics in self.results['class_metrics'].items():
            classes.append(label)
            precision_values.append(metrics['precision'])
            recall_values.append(metrics['recall'])
            f1_values.append(metrics['f1'])

        # Create plot
        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision_values, width, label='Precision', color='#5DA5DA')
        ax.bar(x, recall_values, width, label='Recall', color='#FAA43A')
        ax.bar(x + width, f1_values, width, label='F1 Score', color='#60BD68')

        ax.set_ylabel('Score')
        ax.set_title('Metrics by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add values on top of bars
        for i, v in enumerate(precision_values):
            ax.text(i - width, v + 0.01, f'{v:.2f}', ha='center')
        for i, v in enumerate(recall_values):
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
        for i, v in enumerate(f1_values):
            ax.text(i + width, v + 0.01, f'{v:.2f}', ha='center')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def plot_bootstrap_confidence_intervals(self, save_path: Optional[str] = None):
        """
        Plot confidence intervals from bootstrap analysis.

        Args:
            save_path: Path to save the figure (if None, will show the figure)
        """
        if not self.bootstrap_results:
            raise ValueError("Must run bootstrap_confidence_intervals() before plotting")

        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_means = []
        ci_lowers = []
        ci_uppers = []

        # Get accuracy
        metric_means.append(self.bootstrap_results['accuracy']['mean'])
        ci_lowers.append(self.bootstrap_results['accuracy']['ci_lower'])
        ci_uppers.append(self.bootstrap_results['accuracy']['ci_upper'])

        # Get weighted metrics
        for metric in ['precision', 'recall', 'f1']:
            metric_means.append(self.bootstrap_results['weighted_metrics'][metric]['mean'])
            ci_lowers.append(self.bootstrap_results['weighted_metrics'][metric]['ci_lower'])
            ci_uppers.append(self.bootstrap_results['weighted_metrics'][metric]['ci_upper'])

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metrics))

        # Plot means and confidence intervals
        ax.errorbar(
            x,
            metric_means,
            yerr=[
                np.array(metric_means) - np.array(ci_lowers),
                np.array(ci_uppers) - np.array(metric_means)
            ],
            fmt='o',
            capsize=5,
            ecolor='#888888',
            marker='s',
            mfc='#4CAF50',
            mec='black',
            ms=10
        )

        # Add annotations
        for i, (mean, lower, upper) in enumerate(zip(metric_means, ci_lowers, ci_uppers)):
            ax.annotate(
                f'{mean:.3f}\n[{lower:.3f}, {upper:.3f}]',
                xy=(i, mean),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
            )

        # Customize plot
        ax.set_ylabel('Value')
        ax.set_title(f'Bootstrap Metrics with {self.bootstrap_results["parameters"]["confidence_level"]*100}% Confidence Intervals')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1.1)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def save_results(self, output_dir: str):
        """
        Save all evaluation results to JSON files.

        Args:
            output_dir: Directory to save results
        """
        import os

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save main evaluation results if available
        if self.results:
            with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
                json.dump(self.results, f, indent=2)

        # Save bootstrap results if available
        if self.bootstrap_results:
            with open(os.path.join(output_dir, 'bootstrap_results.json'), 'w') as f:
                json.dump(self.bootstrap_results, f, indent=2)

        # Save cross-validation results if available
        if self.cross_val_results:
            with open(os.path.join(output_dir, 'cross_validation_results.json'), 'w') as f:
                json.dump(self.cross_val_results, f, indent=2)

        print(f"Results saved to {output_dir}")

    def predict_with_probabilities(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with probabilities for a list of financial texts.

        Args:
            texts: List of financial text samples to classify

        Returns:
            Tuple of (predictions, probabilities) as numpy arrays
        """
        predictions = []
        probabilities = []

        # Process in batches
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            batch_probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            batch_preds = np.argmax(batch_probs, dim=1)

            predictions.extend(batch_preds)
            probabilities.extend(batch_probs)

        return np.array(predictions), np.array(probabilities)

    def calibration_analysis(
        self,
        texts: List[str],
        true_labels: List[Union[int, str]],
        n_bins: int = 10
    ) -> Dict:
        """
        Perform calibration analysis to assess reliability of prediction probabilities.

        Args:
            texts: List of financial text samples
            true_labels: Ground truth labels
            n_bins: Number of probability bins for reliability diagram

        Returns:
            Dictionary with calibration analysis results
        """
        # Convert string labels to indices if needed
        if isinstance(true_labels[0], str):
            label_map = {label: i for i, label in enumerate(self.labels)}
            true_labels = [label_map.get(label, -1) for label in true_labels]

        # Get predictions with probabilities
        _, probabilities = self.predict_with_probabilities(texts)

        # Extract confidence (max probability) and predictions
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)

        # Convert true labels to binary indicators per class
        true_one_hot = np.zeros((len(true_labels), len(self.labels)))
        for i, label in enumerate(true_labels):
            true_one_hot[i, label] = 1

        # Calculate expected calibration error (ECE)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Ensure valid bin indices

        bin_accuracies = np.zeros(n_bins)
        bin_confidences = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            bin_mask = bin_indices == i
            if np.sum(bin_mask) > 0:
                bin_accuracies[i] = np.mean(predictions[bin_mask] == np.array(true_labels)[bin_mask])
                bin_confidences[i] = np.mean(confidences[bin_mask])
                bin_counts[i] = np.sum(bin_mask)

        # Calculate ECE
        ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / np.sum(bin_counts)

        # Calculate Maximum Calibration Error (MCE)
        mce = np.max(np.abs(bin_accuracies - bin_confidences))

        # Calculate Brier score
        brier_scores = []
        for i, label in enumerate(true_labels):
            brier_scores.append(np.sum((probabilities[i] - true_one_hot[i])**2))
        brier_score = np.mean(brier_scores)

        # Prepare calibration results
        calibration_results = {
            'expected_calibration_error': float(ece),
            'maximum_calibration_error': float(mce),
            'brier_score': float(brier_score),
            'bin_details': {
                'bin_edges': bin_edges.tolist(),
                'bin_accuracies': bin_accuracies.tolist(),
                'bin_confidences': bin_confidences.tolist(),
                'bin_counts': bin_counts.tolist()
            }
        }

        return calibration_results

    def plot_reliability_diagram(self, calibration_results: Dict, save_path: Optional[str] = None):
        """
        Plot reliability diagram from calibration analysis.

        Args:
            calibration_results: Results from calibration_analysis
            save_path: Path to save the figure (if None, will show the figure)
        """
        bin_accuracies = np.array(calibration_results['bin_details']['bin_accuracies'])
        bin_confidences = np.array(calibration_results['bin_details']['bin_confidences'])
        bin_counts = np.array(calibration_results['bin_details']['bin_counts'])
        bin_edges = np.array(calibration_results['bin_details']['bin_edges'])

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

        # Plot bin accuracy vs. confidence
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        valid_bins = bin_counts > 0

        # Plot actual calibration points
        ax.bar(
            bin_centers,
            bin_accuracies - bin_confidences,
            width=(bin_edges[1] - bin_edges[0]) * 0.9,
            alpha=0.3,
            color='#FF5722',
            label='Calibration Gap'
        )

        # Plot bins with accuracy and confidence
        ax.scatter(
            bin_confidences[valid_bins],
            bin_accuracies[valid_bins],
            s=bin_counts[valid_bins] * 100 / np.max(bin_counts),
            alpha=0.8,
            c='#2196F3',
            label='Model Calibration'
        )

        # Add ECE and Brier score
        ece = calibration_results['expected_calibration_error']
        brier = calibration_results['brier_score']
        ax.text(
            0.05,
            0.95,
            f"ECE: {ece:.4f}\nBrier Score: {brier:.4f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        # Customize plot
        ax.set_xlabel('Confidence (Predicted Probability)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Reliability Diagram')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def domain_specific_evaluation(
        self,
        domain_datasets: Dict[str, Tuple[List[str], List[Union[int, str]]]],
    ) -> Dict:
        """
        Evaluate model performance across different financial text domains.

        Args:
            domain_datasets: Dictionary mapping domain names to (texts, labels) tuples

        Returns:
            Dictionary with domain-specific evaluation results
        """
        domain_results = {}

        for domain_name, (texts, labels) in domain_datasets.items():
            print(f"Evaluating on domain: {domain_name} ({len(texts)} samples)")

            # Evaluate on this domain
            domain_eval = self.evaluate(texts, labels)

            # Store key metrics
            domain_results[domain_name] = {
                'sample_count': len(texts),
                'accuracy': domain_eval['accuracy'],
                'weighted_metrics': domain_eval['weighted_metrics'],
                'class_distribution': {
                    label: domain_eval['class_metrics'][label]['support']
                    for label in self.labels
                }
            }

        # Calculate relative performance
        all_accuracies = [results['accuracy'] for results in domain_results.values()]
        all_f1s = [results['weighted_metrics']['f1'] for results in domain_results.values()]

        avg_accuracy = np.mean(all_accuracies)
        avg_f1 = np.mean(all_f1s)

        # Add relative performance for each domain
        for domain_name in domain_results:
            domain_results[domain_name]['relative_performance'] = {
                'accuracy_vs_avg': domain_results[domain_name]['accuracy'] / avg_accuracy,
                'f1_vs_avg': domain_results[domain_name]['weighted_metrics']['f1'] / avg_f1
            }

        # Add overall summary
        domain_evaluation = {
            'domain_results': domain_results,
            'summary': {
                'average_accuracy': avg_accuracy,
                'average_f1': avg_f1,
                'std_accuracy': np.std(all_accuracies),
                'std_f1': np.std(all_f1s),
                'best_domain': max(domain_results.keys(), key=lambda k: domain_results[k]['accuracy']),
                'worst_domain': min(domain_results.keys(), key=lambda k: domain_results[k]['accuracy'])
            }
        }

        return domain_evaluation

    def plot_domain_comparison(self, domain_evaluation: Dict, save_path: Optional[str] = None):
        """
        Plot performance comparison across different domains.

        Args:
            domain_evaluation: Results from domain_specific_evaluation
            save_path: Path to save the figure (if None, will show the figure)
        """
        domains = list(domain_evaluation['domain_results'].keys())
        accuracies = [domain_evaluation['domain_results'][d]['accuracy'] for d in domains]
        f1_scores = [domain_evaluation['domain_results'][d]['weighted_metrics']['f1'] for d in domains]

        # Sort domains by accuracy
        sorted_indices = np.argsort(accuracies)[::-1]
        domains = [domains[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(domains))
        width = 0.35

        ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#3F51B5')
        ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='#009688')

        # Add average lines
        ax.axhline(
            y=domain_evaluation['summary']['average_accuracy'],
            linestyle='--',
            color='#3F51B5',
            alpha=0.7,
            label='Avg Accuracy'
        )
        ax.axhline(
            y=domain_evaluation['summary']['average_f1'],
            linestyle='--',
            color='#009688',
            alpha=0.7,
            label='Avg F1'
        )

        # Customize plot
        ax.set_ylabel('Score')
        ax.set_title('Performance Across Financial Text Domains')
        ax.set_xticks(x)
        ax.set_xticklabels(domains, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add values on top of bars
        for i, v in enumerate(accuracies):
            ax.text(i - width/2, v + 0.01, f'{v:.2f}', ha='center')
        for i, v in enumerate(f1_scores):
            ax.text(i + width/2, v + 0.01, f'{v:.2f}', ha='center')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def statistical_significance_test(
        self,
        texts: List[str],
        true_labels: List[Union[int, str]],
        benchmark_predictions: Dict[str, List[int]],
        alpha: float = 0.05
    ) -> Dict:
        """
        Perform statistical significance tests comparing model performance with benchmarks.

        Args:
            texts: List of financial text samples
            true_labels: Ground truth labels
            benchmark_predictions: Dictionary mapping model names to their predictions
            alpha: Significance level for statistical tests

        Returns:
            Dictionary with significance test results
        """
        # Convert string labels to indices if needed
        if isinstance(true_labels[0], str):
            label_map = {label: i for i, label in enumerate(self.labels)}
            true_labels = [label_map.get(label, -1) for label in true_labels]

        # Get our model's predictions
        our_predictions = self.predict(texts)

        # Initialize results dictionary
        significance_results = {
            'parameters': {
                'alpha': alpha,
                'sample_size': len(texts)
            },
            'model_comparisons': {}
        }

        # Generate metrics for our model
        our_accuracy = accuracy_score(true_labels, our_predictions)

        # Compare with each benchmark model
        for model_name, bench_preds in benchmark_predictions.items():
            # Calculate benchmark accuracy
            bench_accuracy = accuracy_score(true_labels, bench_preds)

            # Prepare contingency table for McNemar's test
            # [both correct, our correct / bench wrong]
            # [our wrong / bench correct, both wrong]
            contingency = np.zeros((2, 2), dtype=int)

            our_correct = (our_predictions == true_labels)
            bench_correct = (bench_preds == true_labels)

            contingency[0, 0] = np.sum(our_correct & bench_correct)
            contingency[0, 1] = np.sum(our_correct & ~bench_correct)
            contingency[1, 0] = np.sum(~our_correct & bench_correct)
            contingency[1, 1] = np.sum(~our_correct & ~bench_correct)

            # Perform McNemar's test
            try:
                mcnemar_result = stats.mcnemar(contingency, correction=True)
                p_value = mcnemar_result.pvalue
                chi2 = mcnemar_result.statistic
                significant = p_value < alpha
            except ValueError:
                # Handle case where McNemar's test cannot be performed
                p_value = 1.0
                chi2 = 0.0
                significant = False

            # Store results
            significance_results['model_comparisons'][model_name] = {
                'our_accuracy': float(our_accuracy),
                'benchmark_accuracy': float(bench_accuracy),
                'accuracy_difference': float(our_accuracy - bench_accuracy),
                'mcnemar_test': {
                    'chi2': float(chi2),
                    'p_value': float(p_value),
                    'significant': significant,
                    'contingency_table': contingency.tolist()
                }
            }

        return significance_results

# Example usage functions
def load_financial_dataset(path, format='csv'):
    """
    Load a financial text dataset from a file.

    Args:
        path: Path to the dataset file
        format: File format ('csv', 'json', 'txt')

    Returns:
        Tuple of (texts, labels)
    """
    if format.lower() == 'csv':
        df = pd.read_csv(path)
        # Assuming columns 'text' and 'label' exist
        return df['text'].tolist(), df['label'].tolist()

    elif format.lower() == 'json':
        with open(path, 'r') as f:
            data = json.load(f)
        # Assuming a list of dictionaries with 'text' and 'label' keys
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        return texts, labels

    elif format.lower() == 'txt':
        texts = []
        labels = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    labels.append(parts[0])
                    texts.append(parts[1])
        return texts, labels

    else:
        raise ValueError(f"Unsupported format: {format}")

def process_benchmark_results_file(path):
    """
    Load benchmark results from a file.

    Args:
        path: Path to the benchmark results file (JSON format)

    Returns:
        Dictionary of benchmark results
    """
    with open(path, 'r') as f:
        return json.load(f)

def run_full_evaluation(
    model_name='ProsusAI/finbert',
    test_data_path='wsb_enhanced_analysis.csv',
    output_dir='finbert_evaluation_results',
    run_bootstrap=True,
    run_cross_val=True
):
    """
    Run a full evaluation pipeline for finBERT.

    Args:
        model_name: HuggingFace model identifier
        test_data_path: Path to test dataset
        output_dir: Directory to save results
        run_bootstrap: Whether to run bootstrap analysis
        run_cross_val: Whether to run cross-validation
    """
    print(f"Loading model: {model_name}")
    evaluator = FinBERTEvaluator(model_name=model_name)

    print(f"Loading test data from: {test_data_path}")
    texts, labels = load_financial_dataset(test_data_path)
    print(f"Loaded {len(texts)} test samples")

    # Basic evaluation
    print("Running basic evaluation...")
    results = evaluator.evaluate(texts, labels)

    # Print summary
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print("Class metrics:")
    for label, metrics in results['class_metrics'].items():
        print(f"  {label}: Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, "
              f"Support={metrics['support']}")

    print("\nWeighted metrics:")
    for metric, value in results['weighted_metrics'].items():
        print(f"  {metric}: {value:.4f}")

    # Error analysis
    print("\nRunning error analysis...")
    error_results = evaluator.error_analysis(texts, labels)
    print(f"Error rate: {error_results['error_rate']:.4f}")
    print("Error rate by class:")
    for label, rate in error_results['error_rate_by_class'].items():
        print(f"  {label}: {rate:.4f}")

    # Bootstrap analysis (optional)
    if run_bootstrap:
        print("\nRunning bootstrap confidence intervals (this may take a while)...")
        bootstrap_results = evaluator.bootstrap_confidence_intervals(texts, labels)
        print(f"Bootstrap accuracy: {bootstrap_results['accuracy']['mean']:.4f} "
              f"[{bootstrap_results['accuracy']['ci_lower']:.4f}, "
              f"{bootstrap_results['accuracy']['ci_upper']:.4f}]")

    # Cross-validation (optional)
    if run_cross_val:
        print("\nRunning cross-validation (this may take a while)...")
        cv_results = evaluator.cross_validate(texts, labels)
        print(f"Cross-validation accuracy: {cv_results['fold_statistics']['accuracy']['mean']:.4f} "
              f"± {cv_results['fold_statistics']['accuracy']['std']:.4f}")

    # Save results
    print(f"\nSaving results to: {output_dir}")
    evaluator.save_results(output_dir)

    # Plot confusion matrix
    print("Generating visualizations...")
    evaluator.plot_confusion_matrix(save_path=f"{output_dir}/confusion_matrix.png")
    evaluator.plot_metrics_comparison(save_path=f"{output_dir}/metrics_comparison.png")

    if run_bootstrap:
        evaluator.plot_bootstrap_confidence_intervals(save_path=f"{output_dir}/bootstrap_ci.png")

    print("Evaluation complete!")

if __name__ == "__main__":
    # Example usage
    run_full_evaluation()
