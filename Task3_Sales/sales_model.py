import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("sales_dataset.csv")

print(df.head())

# Features and target
X = df[['TV','Radio','Newspaper']]
y = df['Sales']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Model
model = LinearRegression()

# Train model
model.fit(X_train,y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test,y_pred)

print("Mean Squared Error:", mse)
