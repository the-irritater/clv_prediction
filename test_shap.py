import shap
import numpy as np
import matplotlib.pyplot as plt

X = np.random.randn(100, 5)
y = X[:, 0] + X[:, 1]
from sklearn.linear_model import Ridge
model = Ridge().fit(X, y)
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

fig, ax = plt.subplots(figsize=(12, 8))
shap.plots.beeswarm(shap_values, show=False)
plt.savefig('test_shap1.png')
plt.close(fig)
