from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import pandas as pd

# Convert your data to a DataFrame
df = pd.read_csv('Application.csv')
# Selecting numerical columns to scale
numerical_columns = [col for col in df.columns if col not in ['machine', 'app','REALTIME (sec)']]

# Initialize and fit the StandardScaler on the numerical features of the training set
scaler = StandardScaler()
scaler.fit(df[numerical_columns])

# Scale the features
df_scaled = df.copy()
df_scaled[numerical_columns] = scaler.transform(df[numerical_columns])
numerical_means = df_scaled[numerical_columns].mean(axis=0)

# Fill missing values in the scaled data with the mean values
df_scaled[numerical_columns] = df_scaled[numerical_columns].fillna(numerical_means)
print(df_scaled[numerical_columns].shape)
# Fit the TabNet model (assuming binary classification for the sake of this example)
clf = TabNetClassifier(n_d=128, n_a=128, n_steps=10, n_independent=4, n_shared=2)
threshold = 50
# Convert the target variable into binary format
target = (df['REALTIME (sec)'] > threshold).astype(int)

clf.fit(df_scaled[numerical_columns].values, target, max_epochs=10)

# Function to extract embeddings for each row (instance)
# Assuming clf is the trained TabNetClassifier and df_scaled is your scaled DataFrame
def extract_embeddings(model, data):
    """Extract embeddings from the model for each instance (row)."""
    model.network.eval()  # Set the model to evaluation mode
    embeddings = []
    for instance in data:
        with torch.no_grad():
            # Reshape the instance to meet the expected input dimensions of the model
            instance = instance.reshape(1, -1)
            instance_tensor = torch.from_numpy(instance).float()
            # Forward pass through the model up to the penultimate layer
            _, batch_embeddings = model.network(instance_tensor)
            # Ensure that batch_embeddings is correctly reshaped
            batch_embeddings = batch_embeddings.cpu().numpy().reshape(-1) 
            embeddings.append(batch_embeddings)
    # Concatenate all embeddings
    embeddings = np.array(embeddings) # This line is updated to use np.array for correct shape handling
    return embeddings

# Extract embeddings for the scaled data
print(df_scaled[numerical_columns].values.shape)
print('===')
embeddings = extract_embeddings(clf, df_scaled[numerical_columns].values)

# Show the extracted embeddings
print(embeddings)
print(embeddings.shape)