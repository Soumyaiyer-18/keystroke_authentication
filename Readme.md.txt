# Keystroke Dynamics Authentication using Ensemble Learning and R-RCM

Reproduction of "Ensemble Learning for Authentication Using Keystroke Dynamics and Robust Recurrent Confidence Model" by Kiyani et al. (2020)

**Course:** CS5100 Foundations of Artificial Intelligence, Spring 2026
**Author:** Soumya Iyer
**Institution:** Northeastern University

## Overview

This project reproduces the core contributions of Kiyani et al.'s keystroke dynamics authentication system:
- Feature extraction pipeline (7 timing features)
- Ensemble learning with three diverse classifiers (SVM, Random Forest, XGBoost)
- Robust Recurrent Confidence Model (R-RCM) with two-threshold adaptive detection
- ANGA/ANIA evaluation metrics

## Results

| Metric | Result | Meaning |
|---|---|---|
| Ensemble accuracy | 98.12% | Overall classification accuracy |
| ANGA | 100% | Genuine user is never locked out (usability) |
| ANIA | 4% | Average share of keystrokes before an impostor is detected (lower is better) |

## Installation

Requires Python 3.8+ and pip.

```bash
git clone https://github.com/Soumyaiyer-18/keystroke-authentication-capstone.git
cd keystroke-authentication-capstone
pip install -r requirements.txt
```

## Usage

Run the complete demo:
```bash
python capstone_milestone3/demo_notebook.py
```

Or open the Jupyter notebook:
```bash
jupyter notebook capstone_milestone3/milestone3_demo.ipynb
```
