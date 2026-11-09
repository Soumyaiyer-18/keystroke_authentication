# Keystroke Dynamics Authentication using Ensemble Learning and R-RCM

Reproduction of "Ensemble Learning for Authentication Using Keystroke Dynamics and Robust Recurrent Confidence Model" by Kiyani et al. (2020)

**Course:** CS5100 Foundations of Artificial Intelligence  
**Semester:** Spring 2026  
**Student:** Soumya Iyer
**Institution:** Northeastern University

## Overview

This project reproduces the core contributions of Kiyani et al.'s keystroke dynamics authentication system:
- Feature extraction pipeline (7 timing features)
- Ensemble learning with three diverse classifiers (SVM, Random Forest, XGBoost)
- Robust Recurrent Confidence Model (R-RCM) with two-threshold adaptive detection
- ANGA/ANIA evaluation metrics

## Results

- **Ensemble Accuracy:** 98.12%
- **ANGA (Genuine User Usability):** 100%
- **ANIA (Impostor Detection Speed):** 4%

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/keystroke-authentication-capstone.git
cd keystroke-authentication-capstone
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Demo
```bash
python capstone_milestone3/demo_notebook.py
```

### Run Jupyter Notebook
```bash
jupyter notebook capstone_milestone3/milestone3_demo.ipynb
```

## Project Structure
