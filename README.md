This repository provides the Python implementation for VLDB'25 paper [Calibrating Noise for Group Privacy in Subsampled Mechanisms](https://drive.google.com/file/d/1cmJ_vlbWMmFqGeUVdtrBBTe_dayjbol9/view).

# Calibrating Noise for Group Privacy

## Requirements 
- torch, torchvision
- kymatio
- numpy
- scipy
- jupyter
- autodp (https://github.com/yuxiangw/autodp)

## Files
> noise_scale_bounds: calibrating noise scale for subsampled Gaussian, Laplace, Skellam, and RR mechanisms. 

> model_training: training machine learning models by DP-SGD algorithm with group privacy guarantees.

> model_training/calibrate_sgm_noise.ipynb: given privacy parameters (i.e., $\epsilon$, $\delta$, and group size), sampling rate, and number of iterations, calibrating the required Gaussian noise for the DP-SGD algorithm.

> model_training/train_\$dataset-name\$.ipynb: code for training models with GP guarantees using DP-SGD algorithm.

> MLModel.py: machine learning models for MNIST, FEMNIST, and CIFAR-10 datasets.

## Usage and Examples
###  Numerical comparison of RGP bounds
- Run Jupyter notebook ```noise_scale_bounds/calibrate_$mechanism$.ipynb```
- Key parameters:
```python
# code segment in calibrate_$mechanism$.ipynb
q = 0.03         # sampling rate
iters = 500      # number of iterations
m = 16           # group size
alpha = 4        # the order of Rényi divergence
tau = 1          # the privacy budget of (m, alpha, tau)-RGP
```
These notebooks are used to calculate and plot RGP bounds (i.e., our bound, the baseline solution, and the analytical lower bound) across different configurations of the Gaussian, Laplace, Skellam, and RR mechanisms.

### DP-SGD with RGP guarantees
1. Given a set of privacy parameters and the mechanism's hyper-parameters, the required noise can be calibrated using binary search by running the following Jupyter notebook ```model_training/calibrate_sgm_noise.ipynb```.
2. Run Jupyter notebook ```model_training/train_$dataset-name$.ipynb```.

A key component in ```calibrate_sgm_noise.ipynb``` is the following function implemented in ```privacy_analysis.py```:
```python
"""
This function calibrates the required noise based on the configuration of the subsampled mechanism as follows:
q: sampling rate
epsilon, m, delta: privacy parameters, i.e.,  (m,epsilon,delta)-GP
t: number of iterations
err: error tolerance for binary search for calibrating noise std
""" 
calibrating_sgm_noise_rdp(q, epsilon, m, delta=1e-5, t=500, err=1e-3)
```

Key parameters in ```train_$dataset-name$.ipynb```:
```python
configs = {
    'output_size': 10,        # number of units in output layer
    'data_size': data_size,   # size of training dataset
    'model': 'scatter',       # model type:
    'data': d,                # loaded dataset, e.g., d=load_mnist()
    'lr': lr,                 # learning rate
    'E': 500,                 # number of SGD iterations
    'q': 0.05,                # sampling rate
    'clip': 0.1,              # L_2 clipping norm
    'sigma': 19.09            # noise scale for RGP
}
```
These notebook are used to train logistic regression (LR) and convolutional neural networks (CNN) paired with Scattering Networks (SN) on (Fashion)MNIST and CIFAR-10 datasets with group privacy guarantees. All models in our experiments are trained using the standard DP-SGD algorithm.
