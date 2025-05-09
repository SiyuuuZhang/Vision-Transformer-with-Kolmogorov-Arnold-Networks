# Vision Transformer with Kolmogorov-Arnold Networks

## **Overview**
This repository implements **VisionKAN**, an innovative model that replaces the traditional Multi-Layer Perceptron (MLP) in Vision Transformer (ViT) with Kolmogorov-Arnold Networks (KAN). By leveraging KAN's learnable activation functions, VisionKAN achieves improved performance in terms of accuracy, stability, and parameter efficiency compared to standard ViT.

## **Key Contributions**
- **Hybrid Architecture**: Integrates KAN into the Transformer encoder, replacing the fixed activation functions in MLP with learnable B-spline-based activations.
- **Efficient Implementation**: Optimized for small-scale models (e.g., batch size=16) to run on consumer-grade GPUs like NVIDIA RTX 3060 Ti.
- **Performance Gains**: Demonstrates superior classification performance on benchmark datasets (e.g., ImageNet, CIFAR-100) with reduced training loss and improved TOP-1 accuracy.

## Methodology
- **Vision Transformer (ViT)**:
  - Image patches are embedded with positional encoding and processed through a standard Transformer encoder.
  - Traditional MLP blocks in the encoder are replaced with KAN modules.
- **Kolmogorov-Arnold Networks (KAN)**:
  - Utilizes learnable activation functions on edges (B-spline parametrization) instead of fixed node activations.
  - Enhances model flexibility, interpretability, and parameter efficiency.

## Reference
- [1] Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (2021).
- [2] Liu et al. *KAN: Kolmogorov-Arnold Networks* (2024).