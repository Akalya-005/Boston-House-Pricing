import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Income vs Price
plt.figure(figsize=(6, 4))
plt.scatter(X['MedInc'], y, alpha=0.5)
plt.xlabel("Median Income")
plt.ylabel("House Price ($100,000s)")
plt.title("Income vs House Price")
plt.grid(True)
plt.tight_layout()
plt.show()
