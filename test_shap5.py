import shap
import numpy as np
import matplotlib.pyplot as plt

X = np.random.randn(10, 5)
y = X[:, 0]
model = __import__('sklearn.linear_model').linear_model.Ridge().fit(X, y)
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

print("Open figures before:", len(plt.get_fignums()))
shap.plots.scatter(shap_values[:, 0], show=False)
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.savefig('test_shap5.png')
plt.close(fig)

print("Open figures after scatter:", len(plt.get_fignums()))

shap.plots.beeswarm(shap_values, show=False)
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.savefig('test_shap6.png')
plt.close(fig)

print("Open figures after beeswarm:", len(plt.get_fignums()))
