# AFT-YOLO Core Modules for Micro-LED Chip Defect Detection

This repository releases part of the code and dataset associated with our paper:

"Joint defect detection for Micro-LED chip arrays based on AFT-YOLO and ultraviolet photoluminescence"

The released code contains three core modules of AFT-YOLO:

- AWD: Adaptive-Weighted Down-sampling module
- FFDN: Feature-Focused Diffusion Network
- TDDH: Task-Aligned Dynamic Detection Head

These modules were designed to be integrated into a YOLOv8s-based detector for high-resolution Micro-LED chip array defect detection. The full detector, training, inference, and deployment procedures follow the YOLOv8 framework and are therefore not repeated in this repository.

A partial Micro-LED defect dataset is also provided for academic reference, visualization, and code testing.

---

## Repository Structure

```text
.
├── AWD.py              # Adaptive-Weighted Down-sampling module
├── FFDN.py             # Feature-Focused Diffusion Network
├── TDDH.py             # Task-Aligned Dynamic Detection Head
│
├── dataset/            # Partial Micro-LED defect dataset
│   ├── images/
│   └── labels/
│
└── README.md