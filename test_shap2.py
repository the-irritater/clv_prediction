import shap
import numpy as np
import matplotlib.pyplot as plt

X = np.random.randn(10, 5)
y = X[:, 0]
model = __import__('sklearn.linear_model').linear_model.Ridge().fit(X, y)
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

fig1, ax = plt.subplots(figsize=(12, 8))
shap.plots.beeswarm(shap_values, show=False)
fig2 = plt.gcf()

print("fig1 == fig2?", fig1 is fig2)
print("Number of open figures:", len(plt.get_fignums()))
