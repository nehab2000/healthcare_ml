"""
Model evaluation utilities.

Evaluates trained models on test or holdout datasets.

Usage:
    # Evaluate on test set
    python evaluation/evaluate.py --model_type cnn --checkpoint checkpoints/cnn/best_model.pth --dataset test
    
    # Evaluate on holdout set (final evaluation)
    python evaluation/evaluate.py --model_type cnn --checkpoint checkpoints/cnn/best_model.pth --dataset holdout

Outputs:
    - Detailed metrics (accuracy, F1, AUC, etc.)
    - ROC curve plot
    - Confusion matrix plot
    - JSON file with all results
"""

import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from data.dataloader import get_data_loaders, get_holdout_loader
from training.utils import calculate_metrics


def evaluate_model(model, data_loader, device, criterion=None):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        device: Device to run on
        criterion: Optional loss function
        
    Returns:
        Dictionary of metrics and predictions
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probas = []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            
            probas = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
    
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probas)
    )
    
    if criterion:
        metrics['loss'] = running_loss / len(data_loader)
    
    return {
        'metrics': metrics,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probas
    }


def evaluate_on_holdout(model, data_config, model_config, device, checkpoint_path=None):
    """
    Evaluate model on holdout set.
    
    Args:
        model: Model instance
        data_config: Data configuration
        model_config: Model configuration
        device: Device to run on
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Evaluation results dictionary
    """
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Get holdout loader
    holdout_loader = get_holdout_loader(data_config, model_config)
    
    # Evaluate
    results = evaluate_model(model, holdout_loader, device)
    
    return results


def print_evaluation_report(results, dataset_name="Dataset"):
    """
    Print detailed evaluation report.
    
    Args:
        results: Results dictionary from evaluate_model
        dataset_name: Name of the dataset
    """
    metrics = results['metrics']
    
    print("\n" + "="*60)
    print(f"EVALUATION REPORT: {dataset_name}")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    if 'auc' in metrics:
        print(f"  AUC:       {metrics['auc']:.4f}")
    if 'loss' in metrics:
        print(f"  Loss:      {metrics['loss']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  NORMAL:")
    print(f"    Precision: {metrics['precision_normal']:.4f}")
    print(f"    Recall:    {metrics['recall_normal']:.4f}")
    print(f"    F1-Score:  {metrics['f1_normal']:.4f}")
    
    print(f"  PNEUMONIA:")
    print(f"    Precision: {metrics['precision_pneumonia']:.4f}")
    print(f"    Recall:    {metrics['recall_pneumonia']:.4f}")
    print(f"    F1-Score:  {metrics['f1_pneumonia']:.4f}")
    
    # Confusion matrix
    if 'confusion_matrix' in metrics:
        cm = np.array(metrics['confusion_matrix'])
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              NORMAL  PNEUMONIA")
        print(f"Actual NORMAL    {cm[0,0]:4d}      {cm[0,1]:4d}")
        print(f"      PNEUMONIA  {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    print("="*60)


def save_evaluation_results(results, output_path):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Path to save results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    save_dict = {
        'metrics': results['metrics'],
        'predictions': [int(p) for p in results['predictions']],
        'labels': [int(l) for l in results['labels']],
        'probabilities': [[float(prob) for prob in probs] for probs in results['probabilities']]
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_dict, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def plot_roc_curve(results, output_path=None):
    """
    Plot ROC curve.
    
    Args:
        results: Results dictionary
        output_path: Optional path to save plot
    """
    labels = np.array(results['labels'])
    probas = np.array(results['probabilities'])
    
    # Get probabilities for positive class
    if probas.ndim > 1 and probas.shape[1] > 1:
        y_scores = probas[:, 1]
    else:
        y_scores = probas
    
    fpr, tpr, _ = roc_curve(labels, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {output_path}")
    
    plt.close()


def plot_confusion_matrix(results, output_path=None):
    """
    Plot confusion matrix.
    
    Args:
        results: Results dictionary
        output_path: Optional path to save plot
    """
    cm = np.array(results['metrics']['confusion_matrix'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    
    plt.close()


def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model_type', type=str, choices=['cnn', 'vit'], required=True,
                       help='Model type: cnn or vit')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, choices=['test', 'holdout'], default='test',
                       help='Dataset to evaluate on')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load configurations
    with open('config/data_config.yaml', 'r') as f:
        data_config = yaml.safe_load(f)['data']
    
    with open('config/model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)['model']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    if args.model_type == 'cnn':
        from models.cnn_model import create_cnn_model
        model = create_cnn_model(model_config)
    else:
        from models.vit_model import create_vit_model
        model = create_vit_model(model_config)
    
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Get data loader
    if args.dataset == 'test':
        _, _, test_loader = get_data_loaders(data_config, model_config)
        loader = test_loader
        dataset_name = "Test Set"
    else:
        loader = get_holdout_loader(data_config, model_config)
        dataset_name = "Holdout Set"
    
    # Evaluate
    results = evaluate_model(model, loader, device)
    
    # Print report
    print_evaluation_report(results, dataset_name)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_evaluation_results(
        results,
        output_dir / f'{args.model_type}_{args.dataset}_results.json'
    )
    
    plot_roc_curve(
        results,
        output_dir / f'{args.model_type}_{args.dataset}_roc.png'
    )
    
    plot_confusion_matrix(
        results,
        output_dir / f'{args.model_type}_{args.dataset}_cm.png'
    )
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

