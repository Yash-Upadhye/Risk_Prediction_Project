import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Simulate financial transaction data
np.random.seed(42)
data_size = 1000
data = pd.DataFrame({
    'transaction_amount': np.random.normal(100000, 25000, data_size),
    'credit_rating': np.random.choice([1, 2, 3, 4, 5], data_size),  # 1: High risk, 5: Low risk
    'interest_rate': np.random.normal(0.05, 0.01, data_size),
    'volatility': np.random.normal(0.2, 0.05, data_size),
    'default': np.random.choice([0, 1], data_size, p=[0.9, 0.1])  # 1: Default, 0: No default
})

# Splitting data
X = data[['transaction_amount', 'credit_rating', 'interest_rate', 'volatility']]
y = data['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Visualizing the distribution of risk exposure
plt.hist(data['credit_rating'], bins=5, alpha=0.7, label='Credit Rating')
plt.title('Distribution of Credit Ratings')
plt.xlabel('Credit Rating')
plt.ylabel('Frequency')
plt.show()
