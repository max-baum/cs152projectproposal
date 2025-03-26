# Methods Outline

We have an existing dat set which has a very large number of negatives (fraud has not been charged) and few positive data points (fraud has been charged). 

We plan to do some preprocessing we will clean, normalize, and design engineered features such as loan amount, business size, location, and NAICS codes to prepare for training.

We plan to explore one class neural network techniques for anomaly detection, treating known fraud cases as the positive class and the rest as unlabeled.

Since we have the lack of true negatives, we used semi-supervised metrics such as precision-at-K, silhouette score, and anomaly ranking as potential in between techniques.

We also plan to use simple feedforward neural networks which we can implement in PyTorch, and/ or PyTorch, TensorFlow/Keras, scikit-learn, pandas as additional tools.

We have some challenges which include data imbalance and potential hidden frauds in the unlabeled set. We addressed this with regularization, dropout, and cross-validation.

We also plan to use dimensionality reduction techniques like t-SNE and UMAP to visualize how well fraud cases were clustered or separated.
