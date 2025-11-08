"""Visualization utilities for model interpretability."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
from typing import List, Tuple, Optional


def visualize_predictions(model, data_loader, device, num_samples=16, output_path=None):
    """
    Visualize sample predictions with confidence scores.
    
    Args:
        model: Trained model
        data_loader: DataLoader
        device: Device
        num_samples: Number of samples to visualize
        output_path: Path to save visualization
    """
    model.eval()
    class_names = ['NORMAL', 'PNEUMONIA']
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    count = 0
    with torch.no_grad():
        for images, labels in data_loader:
            if count >= num_samples:
                break
            
            images = images.to(device)
            outputs = model(images)
            probas = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            for i in range(min(len(images), num_samples - count)):
                img = images[i].cpu()
                label = labels[i].item()
                pred = preds[i].item()
                prob = probas[i][pred].item()
                
                # Convert tensor to numpy for visualization
                if img.shape[0] == 3:
                    img_np = img.permute(1, 2, 0).numpy()
                    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                    img_np = np.clip(img_np, 0, 1)
                else:
                    img_np = img.squeeze().numpy()
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                
                axes[count].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
                axes[count].set_title(
                    f'True: {class_names[label]}\n'
                    f'Pred: {class_names[pred]} ({prob:.2f})',
                    color='green' if label == pred else 'red'
                )
                axes[count].axis('off')
                count += 1
    
    # Hide unused subplots
    for i in range(count, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.close()


def visualize_gradcam(model, image, label, device, output_path=None):
    """
    Generate Grad-CAM visualization for CNN models.
    
    Args:
        model: Trained CNN model
        image: Input image tensor
        label: True label
        device: Device
        output_path: Path to save visualization
    """
    model.eval()
    
    # Register hook to get gradients
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Get the last convolutional layer
    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'layer4'):
            target_layer = model.backbone.layer4[-1]
        elif hasattr(model.backbone, 'features'):
            target_layer = model.backbone.features[-1]
        else:
            print("Warning: Could not find target layer for Grad-CAM")
            return
    else:
        print("Warning: Model structure not recognized for Grad-CAM")
        return
    
    # Register hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)
    
    # Forward pass
    image = image.unsqueeze(0).to(device)
    image.requires_grad = True
    output = model(image)
    
    # Backward pass
    model.zero_grad()
    class_idx = output.argmax(dim=1)
    output[0, class_idx].backward()
    
    # Get gradients and activations
    grads = gradients[0]
    acts = activations[0]
    
    # Calculate weights
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    
    # Generate CAM
    cam = torch.sum(weights * acts, dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    img_np = image.squeeze().cpu().detach().numpy()
    if img_np.shape[0] == 3:
        img_np = img_np.transpose(1, 2, 0)
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img_np = np.clip(img_np, 0, 1)
        axes[0].imshow(img_np)
    else:
        img_np = img_np.squeeze()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        axes[0].imshow(img_np, cmap='gray')
    
    axes[0].set_title(f'Original Image (True: {label})')
    axes[0].axis('off')
    
    # Grad-CAM overlay
    axes[1].imshow(img_np.squeeze(), cmap='gray')
    axes[1].imshow(cam, alpha=0.5, cmap='jet')
    axes[1].set_title('Grad-CAM Visualization')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to {output_path}")
    
    plt.close()


def visualize_attention(model, image, device, output_path=None, layer_idx=-1):
    """
    Visualize attention maps for Vision Transformer.
    
    Args:
        model: Trained ViT model
        image: Input image tensor
        device: Device
        output_path: Path to save visualization
        layer_idx: Which attention layer to visualize (-1 for last)
    """
    model.eval()
    
    # Get attention weights
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        attention_weights = model.get_attention_weights(image)
    
    # Get attention from specified layer
    attn = attention_weights[layer_idx][0]  # [num_heads, num_patches+1, num_patches+1]
    
    # Average across heads
    attn = attn.mean(dim=0)
    
    # Get CLS token attention to patches (first row, excluding CLS token)
    cls_attn = attn[0, 1:].cpu().numpy()
    
    # Reshape to image grid (assuming 224x224 image with 16x16 patches = 14x14 grid)
    grid_size = int(np.sqrt(len(cls_attn)))
    attn_map = cls_attn.reshape(grid_size, grid_size)
    
    # Resize to original image size
    attn_map = cv2.resize(attn_map, (224, 224))
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    img_np = image.squeeze().cpu().numpy()
    if img_np.shape[0] == 3:
        img_np = img_np.transpose(1, 2, 0)
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img_np = np.clip(img_np, 0, 1)
        axes[0].imshow(img_np)
    else:
        img_np = img_np.squeeze()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        axes[0].imshow(img_np, cmap='gray')
    
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map
    axes[1].imshow(img_np.squeeze(), cmap='gray')
    axes[1].imshow(attn_map, alpha=0.5, cmap='jet')
    axes[1].set_title('Attention Map (CLS Token)')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to {output_path}")
    
    plt.close()

