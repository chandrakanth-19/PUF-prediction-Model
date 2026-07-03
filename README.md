# PUF Modeling and Evaluation

This repository implements a machine learning pipeline for modeling **Physical Unclonable Functions (PUFs)** using engineered feature mappings and linear classifiers. It also includes utilities for evaluating different learning algorithms, tuning hyperparameters, and reconstructing underlying delay parameters from trained linear models.

## Repository Structure

```text
.
├── puf_model_pipeline.py      # Core implementation
├── experiment_analysis.py     # Model comparison experiments
├── eval.py                    # Evaluation and benchmarking script
├── secret_trn.txt             # Training dataset
├── secret_tst.txt             # Test dataset
├── secret_mod.txt             # Reference models for decoding evaluation
└── README.md
```

## Features

* Custom feature engineering for Arbiter PUF modeling
* Logistic Regression based PUF learning
* Linear and Polynomial SVM comparison
* Hyperparameter tuning over multiple regularization strengths
* Delay parameter reconstruction (model inversion)
* Evaluation of:

  * Classification accuracy
  * Training time
  * Feature mapping time
  * Model reconstruction error

## File Description

### `puf_model_pipeline.py`

Contains the complete modeling pipeline.

Implements:

* `my_map()`

  * Generates engineered parity-product features from challenge bits.

* `my_fit()`

  * Trains a Logistic Regression classifier on transformed features.

* `my_decode()`

  * Reconstructs the underlying delay parameters `(p, q, r, s)` from a learned linear model using constrained linear regression.

---

### `experiment_analysis.py`

Performs comparative experiments between different classifiers.

Evaluated models include:

* Logistic Regression (L1)
* Logistic Regression (L2)
* Linear SVM
* Polynomial Kernel SVM
* Hyperparameter tuning for different values of `C`

Outputs classification accuracies for each configuration.

---

### `eval.py`

Benchmarks the complete pipeline.

Measures:

* Average training time
* Feature mapping time
* Classification error
* Decoding time
* Reconstruction error

Also evaluates different values of the Logistic Regression regularization parameter.

## Methodology

The pipeline follows these steps:

1. Load training and testing CRPs.
2. Transform challenges using handcrafted parity-product features.
3. Train a Logistic Regression classifier.
4. Evaluate prediction accuracy.
5. Decode the learned linear model back into physical delay parameters.
6. Measure reconstruction quality against reference models.

## Requirements

* Python 3.x
* NumPy
* SciPy
* scikit-learn

Install dependencies using:

```bash
pip install numpy scipy scikit-learn
```

## Running Experiments

Run model comparison:

```bash
python experiment_analysis.py
```

Run the complete evaluation pipeline:

```bash
python eval.py
```

## Results

The evaluation scripts report metrics including:

* Classification accuracy
* Training time
* Feature transformation time
* Decoding time
* Model reconstruction error

These metrics can be used to compare different learning algorithms and regularization settings for PUF modeling.

## Authors

Developed as part of a Physical Unclonable Function (PUF) machine learning project implementing feature engineering, model training, evaluation, and delay reconstruction.   
