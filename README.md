# Video2World

> Video2World is a modular 3D reconstruction framework that converts monocular videos into navigable 3D scenes through a modern visual SLAM pipeline. The project is designed with extensibility in mind, allowing different feature extractors, camera pose estimators, mapping strategies, and optimization modules to be integrated independently.

Video2World provides a clean and extensible codebase for:
- camera geometry
- Gaussian splatting rendering
- CUDA acceleration
- depth / video-based models
- dataset loaders
- training & inference pipelines

Built for **research, experimentation, and production-quality engineering**.

---

## ✨ Features

- 📐 Camera geometry (intrinsic, extrinsic, projection)
- ⚡ CUDA Gaussian splatting kernels
- 🧠 Depth & Video2World models
- 📦 YAML config-driven experiments
- 🧪 Professional pytest test suite
- 🧩 Clean `src/` package layout
- 🔬 Easy to extend for new models or datasets

---

## 📦 Installation

### 1. Clone repository
```bash
git clone https://github.com/<your-username>/Video2World.git
cd Video2World
```

### 2. Create environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. (Optional) CUDA build
If using CUDA kernels:
```bash
pip install -v .
```

---

## 🚀 Quick Start

### Run training
```bash
python experiments/train.py --config configs/train.yaml
```

### Run inference
```bash
python experiments/infer.py --config configs/model.yaml
```

### Run tests
```bash
pytest
```

---

## 🧪 Testing

Run all tests:
```bash
pytest
```

Run unit tests only:
```bash
pytest tests/unit
```

Run CUDA tests:
```bash
pytest -m cuda
```

Run coverage:
```bash
pytest --cov=v2w
```

---

## 📁 Project Structure

```
src/v2w
├── geometry      # camera math & projection
├── rendering     # rasterizer & splatting
├── models        # depth + video2world models
├── cuda          # CUDA kernels & bindings
├── datasets      # dataset loaders
├── config        # config system
├── training      # training logic
└── utils         # helpers

configs/           # YAML experiment configs
data/              # small samples / datasets
experiments/       # training & inference scripts
tests/             # pytest suite
notebooks/         # experiments & debugging
```

---

## ⚙️ Configuration

All experiments are driven by YAML configuration files.

Example:
```bash
python experiments/train.py --config configs/train.yaml
```

Available configs:
```
configs/
├── cam.yaml
├── dataset.yaml
├── model.yaml
├── train.yaml
└── default.yaml
```

---

## 📚 Datasets

Currently supported:
- EuRoC
- TartanAir

Place datasets inside:

```
data/
├── euroc/
├── tartanair-v2/
```

---

## 🧠 Example Usage

```python
import torch
from v2w.geometry.projection import project_points

points_3d = torch.randn(100, 3)
K = torch.eye(3)

points_2d = project_points(points_3d, K)
print(points_2d.shape)
```

---

## 🛠 Development

### Format
```bash
black src tests
```

### Lint
```bash
ruff check .
```

### Tests
```bash
pytest
```

