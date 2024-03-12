from tab_transformer_pytorch import TabTransformer
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Example DataFrame
df=pd.read_csv('Machine.csv')

# 1. Separate your categorical and continuous features
categorical_columns = ["machine", "core"]
continuous_columns = [col for col in df.columns if col not in categorical_columns]
df[continuous_columns] = df[continuous_columns].apply(lambda col: col.fillna(col.mean()))
print(categorical_columns)
print(continuous_columns)
print(df.head())

print(df[categorical_columns].shape)
print(df[continuous_columns].shape)

# 2. Encode categorical features
le = LabelEncoder()
df[categorical_columns] = df[categorical_columns].apply(lambda col: le.fit_transform(col))

# 3. Normalize continuous features (if you have any in this step)
scaler = StandardScaler()
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

# Prepare your data for the model
if not df[categorical_columns].empty and not df[continuous_columns].empty:
    x_categ = torch.tensor(df[categorical_columns].values, dtype=torch.long)
    x_cont = torch.tensor(df[continuous_columns].values, dtype=torch.float32)
else:
    print("One of the DataFrames is empty. Check your column names and data preprocessing.")

# 4. Prepare your model (adjust parameters as necessary)
model = TabTransformer(
    categories=[len(df[col].unique()) for col in categorical_columns],  # Number of unique values per categorical column
    num_continuous=len(continuous_columns),
    dim=32,  # Dimensionality of the embeddings
    depth=6,  # Number of transformer blocks
    heads=8,  # Number of attention heads
    dim_head=16,  # Dimensionality of each head
    mlp_hidden_mults=(4, 2),  # Multipliers for the MLP layers
    dim_out=64,  # Output dimensionality of the embeddings
    attn_dropout=0.1,
    ff_dropout=0.1,
)

# 5. Generate embeddings
with torch.no_grad():
    print(f'shape here:{x_categ.shape},x_cont here:{x_cont.shape}')
    embeddings = model(x_categ, x_cont)
    embeddings_df = pd.DataFrame(embeddings.numpy())

    # Save the DataFrame to a CSV file
    embeddings_df.to_csv('embeddings.csv', index=False)
    print(embeddings.shape)
    print(embeddings)

