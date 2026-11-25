import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('sample_data.csv')
data.head()

from sklearn.model_selection import train_test_split

y = data.pop('TARGET CLASS')
x = data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier

error_rates = []

for a in range(1, 40):
    k = a
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    preds = knn.predict(x_test)
    error_rates.append(np.mean(y_test - preds))

plt.figure(figsize=(10, 7))
plt.plot(range(1, 40), error_rates, color='blue', linestyle='dashed', marker='o',
            markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()