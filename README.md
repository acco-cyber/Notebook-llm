#  SHERLOCK: Structured Hierarchical Efficient Resolution-aware Learning with Optimized Class Knowledge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official implementation of **SHERLOCK**, a comprehensive framework for efficient dermoscopic skin lesion classification that achieves clinical-grade accuracy with mobile-friendly computational efficiency. SHERLOCK addresses three critical deployment challenges through:

- **RAAT**: Resolution-Aware Attention Transfer (96.6% FLOPs reduction)
- **DHL**: Dermoscopic Hierarchical Learning (+8.2% melanoma recall)
- **Integrated System**: 90.1% accuracy at only 0.28 GFLOPs

## 📊 Key Results

| Metric | SHERLOCK | Baseline | Improvement |
|--------|----------|----------|-------------|
| Accuracy | 90.1% | 84.2% | +5.9 pp |
| Macro-F1 | 85.8% | 78.4% | +7.4 pp |
| Melanoma Recall | 91.8% | 81.3% | +10.5 pp |
| FLOPs | 0.28 G | 8.35 G | -96.7% |
| Mobile Latency | 89.4 ms | 387.6 ms | -76.9% |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/SHERLOCK.git
cd SHERLOCK

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- scikit-learn
- matplotlib
- opencv-python
- albumentations

### Data Preparation

1. Download HAM10000 dataset from [ISC2017 Challenge](https://challenge.isic-archive.com/data/#2017)
2. Organize the data structure:
```
data/
├── HAM10000/
│   ├── images/
│   │   ├── ISIC_0024306.jpg
│   │   └── ...
│   └── HAM10000_metadata.csv
```

3. Preprocess and split data:
```bash
python scripts/preprocess.py --data_dir data/HAM10000 --output_dir data/processed
```

## 🏗️ Model Architecture

SHERLOCK consists of three integrated components:

### 1. Multi-Model Baseline
Systematic evaluation of 6 architectures (MobileNetV3, EfficientNet, Xception, ResNet50, VGG16/19) to establish performance-efficiency Pareto frontiers.

### 2. Resolution-Aware Attention Transfer (RAAT)
```python
from models.raat import RAAT

# Initialize RAAT with teacher and student models
teacher = XceptionTeacher(resolution=224)
student = EfficientNetStudent(resolution=160)

raat = RAAT(teacher=teacher, student=student, beta=0.3)
raat.train(train_loader, val_loader, epochs=25)
```

### 3. Dermoscopic Hierarchical Learning (DHL)
```python
from models.dhl import DHL

# Initialize DHL with shared backbone and dual heads
model = DHL(backbone='efficientnet-b3', lambda_param=0.5)
model.train(train_loader, val_loader, epochs=20)
```


## 🎯 Training

### Baseline Models
```bash
python train_baseline.py \
  --model efficientnet-b3 \
  --data_dir data/processed \
  --batch_size 32 \
  --epochs 20 \
  --lr 1e-4
```

### RAAT Training
```bash
python train_raat.py \
  --teacher xception \
  --student efficientnet-b0 \
  --teacher_res 224 \
  --student_res 160 \
  --beta 0.3 \
  --data_dir data/processed \
  --epochs 25
```

### DHL Training
```bash
python train_dhl.py \
  --backbone efficientnet-b3 \
  --lambda 0.5 \
  --data_dir data/processed \
  --epochs 20
```

## 📈 Evaluation

```bash
# Evaluate baseline model
python scripts/evaluate.py \
  --model_path checkpoints/baseline_efficientnet-b3.pth \
  --data_dir data/processed/test

# Evaluate SHERLOCK integrated system
python scripts/evaluate.py \
  --model_path checkpoints/sherlock_final.pth \
  --config configs/sherlock.yaml
```

## 📊 Results Reproduction

To reproduce the results from the paper:

1. **Baseline Models**: Run all baseline experiments
```bash
bash scripts/run_baselines.sh
```

2. **RAAT Ablation**: Test different β values
```bash
for beta in 0.1 0.3 0.5 1.0; do
    python train_raat.py --beta $beta
done
```

3. **DHL Ablation**: Test different λ values
```bash
for lambda in 0.3 0.5 0.7; do
    python train_dhl.py --lambda $lambda
done
```

## 📱 Mobile Deployment

SHERLOCK models are optimized for mobile deployment:

```python
# Convert to TorchScript for mobile
model = torch.jit.script(sherlock_model)
torch.jit.save(model, "sherlock_mobile.pt")

# Or export to ONNX
torch.onnx.export(
    sherlock_model,
    dummy_input,
    "sherlock.onnx",
    opset_version=13,
    input_names=['input'],
    output_names=['malignancy', 'subtype']
)
```

## 🩺 Clinical Validation

Generate interpretability maps and clinical reports:

```python
from utils.interpretability import generate_gradcam, clinical_report

# Generate Grad-CAM visualizations
heatmap = generate_gradcam(model, image, class_idx=1)

# Generate clinical report
report = clinical_report(
    predictions=preds,
    confidence_scores=probs,
    metadata=patient_info
)
```


## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Paper](https://arxiv.org/abs/XXXX.XXXXX)
- [Dataset](https://challenge.isic-archive.com/data/#2017)
- [Issue Tracker](https://github.com/your-org/SHERLOCK/issues)

## 🙏 Acknowledgments

We thank the ISIC Archive for providing the HAM10000 dataset and the research community for their valuable open-source contributions.

---

**Disclaimer**: This tool is for research purposes only and is not intended for clinical use without proper validation and regulatory approval.
