# Adversarial disturbances in neural network modeling

This repository accompanies a Master's thesis focused on analyzing the impact of
stochastic and adversarial disturbances on neural network models in image
classification tasks. It includes implementations of experiments, comparative
studies, and the original adversarial algorithm **LESGSM**.

## 🔬 Research highlights

- Analysis of **white noise** and **Cauchy noise** influence on classification
  accuracy
- Comparison of adversarial attacks: **FGSM**, **DeepFool**, and **LESGSM**
- Investigation of the relationship between **perturbation energy** and model
  accuracy
- Evaluation of **noise regularization strategies** to improve robustness
  against adversarial attacks

## 🗃️ Repository structure

- `/attacks` - implementations and tests of adversarial attack algorithms
- `/defences` – defense methods, including noise regularization strategies
- `/models` – model definitions with trained weights
  - `/adv` – models trained with noise layers
  - `/std` – standard (baseline) models
- `/noises` – stochastic noise implementations (e.g., white noise, Cauchy noise)

**Note! ⚠️**

**Naming convention for test files:** all files used in research tests and
reproducibility experiments begin with the prefix `@.` (for example
`@.deepfool.cauchy.ipynb`, `@.lesgsm.normal.ipynb`). This helps quickly identify
files that were part of the thesis evaluations.

## 📃 Requirements

- Python 3.11+
- TensorFlow 2.15+
- Matplotlib, NumPy, pandas, seaborn
- Jupyter Notebook
