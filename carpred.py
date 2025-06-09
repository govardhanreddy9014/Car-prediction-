import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_csv("F:\\GATES\\ML\\Lab\\Car_Purchasing_Data.csv",encoding="latin1")

# Display column names
print(df.columns)

# Drop unnecessary columns if they exist
cols_to_drop = ['Customer Name', 'Customer e-mail']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Encode categorical variables
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])

# Define features (X) and target variable (y)
X = df.drop("Car Purchase Amount", axis=1).values
y = df["Car Purchase Amount"].values

# Split dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Initialize the ANN model
model = Sequential()

# Input Layer + Hidden Layers
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))

# Output Layer
model.add(Dense(units=1))  # No activation function for regression

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Model Summary
model.summary()


# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_mae',    # Monitor validation loss
    patience=3,           # Stop if no improvement for 10 epochs
    restore_best_weights=True  # Keep best model
)

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=100, 
                    batch_size=8, 
                    validation_data=(X_test, y_test), 
                    callbacks=[early_stopping])


# Evaluate performance on test data
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae}")


# Plot Training & Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Model Loss Curve with Early Stopping')
plt.legend()
plt.show()

# Plot Training & Validation MAE
plt.figure(figsize=(8, 5))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Model MAE Curve')
plt.legend()
plt.show()



# Predict on test data
y_pred = model.predict(X_test)

# Compare actual vs. predicted values
df_results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred.flatten()})
print(df_results.head())

# Scatter plot of actual vs. predicted values
plt.scatter(y_test, y_pred, color='blue')
plt.plot(y_test, y_test, color='red')  # Ideal line (Actual = Predicted)
plt.xlabel("Actual Car Purchase Amount")
plt.ylabel("Predicted Car Purchase Amount")
plt.title("Actual vs Predicted Car Purchase Amount")
plt.show()
