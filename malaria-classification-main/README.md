# Malaria Classification using ViT-KANMoE Architecture

## 🔬 Project Overview

This project implements an advanced deep learning model for malaria parasite classification using a novel architecture that combines **Vision Transformer (ViT)** with **Kolmogorov-Arnold Networks (KAN)** and **Mixture of Experts (MoE)**. The model classifies malaria parasites into 7 different classes with high accuracy using state-of-the-art techniques.

## 🏗️ Architecture

### Proposed Model Architecture
<img width="419" alt="Screenshot 2025-06-20 at 4 39 15 PM" src="https://github.com/user-attachments/assets/f8f336c3-5fc3-4a44-ac7f-17b43308800f" />


The ViT-KANMoE architecture consists of three main components working in sequence:

### ViT-KANMoE Model Components

1. **Vision Transformer Backbone**: Pre-trained ViT-B/16 from ImageNet for feature extraction
2. **Kolmogorov-Arnold Networks (KAN)**: Advanced neural network architecture replacing traditional MLPs
3. **Mixture of Experts (MoE)**: Multiple KAN experts with attention-based gating
4. **Attention Gating Network**: Softmax-based expert selection mechanism

### Detailed Architecture Flow

#### 1. Vision Transformer (ViT) Backbone
- **Input Processing**: 224×224 RGB images are split into 16×16 patches
- **Patch Embedding**: Linear projection converts patches to 768-dimensional embeddings
- **Position Encoding**: Learnable positional embeddings added to patch embeddings
- **Self-Attention Layers**: 12 transformer encoder blocks process the sequence
- **CLS Token Extraction**: Global representation extracted from the classification token
- **Transfer Learning**: Only the last 2 transformer layers are fine-tuned

#### 2. Attention Gating Network
- **Input**: 768-dimensional CLS token from ViT backbone
- **Architecture**: 
  - Fully connected layer: 768 → num_experts (8)
  - Dropout layer for regularization
  - Softmax activation for probability distribution
- **Top-K Selection**: Selects top-3 experts based on attention scores
- **Output**: Normalized weights for expert aggregation

#### 3. MoE Module (KAN-based Experts)
- **Expert Architecture**: Each of the 8 experts contains 4 stacked KAN layers
- **KAN Layer Design**:
  - Splits 768-dimensional input into individual scalars
  - Processes each scalar through univariate transformation blocks
  - Uses grouped 1D convolutions for vectorized computation
  - Supports GELU and Mish activation functions
- **Expert Aggregation**: Weighted combination of top-K expert outputs
- **Final Classification**: Linear layer maps aggregated features to 7 classes

### Key Features

- **Fine-tuned ViT**: Only the last 2 layers are trainable for efficient transfer learning
- **Vectorized KAN Layers**: Optimized implementation using grouped convolutions
- **Expert Aggregation**: Parallelized computation with top-k expert selection
- **Advanced Activations**: Support for GELU and Mish activation functions
- **Regularization**: Load balancing and entropy-based regularization for MoE

## 📊 Model Configuration

```python
num_classes = 7       # 7-class malaria classification
num_experts = 8       # Number of KAN experts in MoE
hidden_dim = 256      # Hidden dimension for KAN layers
num_layers = 4        # Number of KAN layers per expert
top_k = 3             # Use top 3 experts per input
reg_lambda = 0.001    # Regularization strength
dropout = 0.2         # Dropout rate
activation = 'GELU'   # Activation function
```

## 🔧 Technical Implementation

### Data Processing
- **Input Size**: 224×224 RGB images (optimized for ViT)
- **Normalization**: Mean=[0.5, 0.5, 0.5], Std=[0.5, 0.5, 0.5]
- **Augmentation**: RandomResizedCrop, RandomHorizontalFlip, ColorJitter
- **Batch Size**: 32 with 4 workers and memory pinning

### Training Configuration
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: CosineAnnealingLR (T_max=10)
- **Loss Function**: CrossEntropyLoss + MoE regularization
- **Gradient Clipping**: max_norm=1.0 for training stability
- **Early Stopping**: Patience=10 epochs
- **Total Epochs**: 200 (with early stopping)

### Advanced Features
- **TensorBoard Logging**: Real-time training/validation loss tracking
- **Model Checkpointing**: Automatic saving of best model based on validation loss
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## 📁 Project Structure

```
malaria-classification/
|-code files/
├-── exp1_5.ipynb           # Main notebook with complete implementation
├─-─ requirements (1).ipynb # Requirements and dependencies notebook
├── results/               # Training results and screenshots
│   ├── Screenshot 2025-06-15 at 7.34.29 PM.png
│   ├── Screenshot 2025-06-15 at 7.34.35 PM.png
│   ├── Screenshot 2025-06-15 at 7.34.46 PM.png
│   └── Screenshot 2025-06-20 at 4.39.15 PM.png  # Architecture diagram
├── .git/                  # Git repository files
├── .gitattributes         # Git attributes configuration
├── .gitignore            # Git ignore rules
├── .DS_Store             # macOS system file
└── README.md             # This documentation file
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision timm numpy scikit-learn matplotlib seaborn tqdm tensorboard
```

### Required Libraries
- PyTorch & TorchVision
- Transformers (timm)
- NumPy
- Scikit-learn
- Matplotlib & Seaborn
- TensorBoard
- tqdm

### Dataset Structure
This project uses a **private malaria dataset** for classification. The model expects data in the following format:
```
data/
├── train_2023/
│   ├── class_0/
│   ├── class_1/
│   ├── ...
│   └── class_6/
└── test_2023/
    ├── class_0/
    ├── class_1/
    ├── ...
    └── class_6/
```

**Note**: The dataset used in this project is proprietary and not publicly available. If you wish to use this model architecture, you can adapt it to work with publicly available malaria datasets or your own dataset following the same directory structure.

### Running the Model

1. **Setup Environment**: Ensure all dependencies are installed
2. **Data Preparation**: Organize your malaria dataset in the required structure
3. **Configure Paths**: Update `train_dir` and `val_dir` in the notebook
4. **Execute Notebook**: Run all cells in sequence
5. **Monitor Training**: Use TensorBoard to track training progress
6. **Evaluate Results**: Check final metrics and confusion matrix

## 📈 Model Performance

The model provides comprehensive evaluation metrics:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and weighted averages
- **Confusion Matrix**: Detailed classification breakdown
- **Training/Validation Loss**: Tracked throughout training

## 🧠 Novel Architecture Details

### Vectorized KAN Layer
- Uses grouped 1D convolutions for efficient computation
- Supports multiple activation functions (GELU, Mish, ReLU)
- Includes dropout for regularization

### Mixture of Experts (MoE)
- 8 specialized KAN experts
- Top-3 expert selection per input
- Attention-based gating with softmax normalization
- Load balancing and entropy regularization

### Transfer Learning Strategy
- Pre-trained ViT-B/16 backbone
- Frozen early layers, trainable last 2 layers
- Custom KANMoE head for classification

## 🔬 Research Applications

This architecture is particularly suitable for:
- Medical image classification
- Fine-grained visual recognition
- Transfer learning scenarios
- Multi-class classification problems
- Research in advanced neural architectures

## 📝 Future Improvements

The notebook includes suggestions for future enhancements:
- Lion and RAdam optimizers
- Focal loss for class imbalance
- OneCycleLR scheduler
- Adaptive attention mechanisms
- Enhanced data augmentation techniques
- Additional KAN layers
- Mish activation optimization

## 📊 Monitoring and Logging

- **TensorBoard Integration**: Real-time loss visualization
- **Progress Bars**: Training progress with tqdm
- **Model Checkpointing**: Automatic best model saving
- **Comprehensive Metrics**: Detailed evaluation reports

## 🤝 Contributing

This project implements cutting-edge research in neural architecture design. Contributions are welcome for:
- Architecture improvements
- Performance optimizations
- Additional evaluation metrics
- Documentation enhancements

## 📄 License

This project is designed for research and educational purposes in medical image analysis and advanced neural network architectures.

---

**Note**: This implementation represents an experimental approach combining multiple state-of-the-art techniques. Results may vary based on dataset characteristics and hyperparameter tuning.
