import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn

def encode_and_embed(df1, df2, embedding_dim=1):
    # Combine 'machine' columns from both dataframes for LabelEncoder
    all_machines = pd.concat([df1['machine'], df2['machine']]).unique()
    le_machine = LabelEncoder()
    le_machine.fit(all_machines)

    # Transform 'machine' columns in both DataFrames
    df1['machine_encoded'] = le_machine.transform(df1['machine'])
    df2['machine_encoded'] = le_machine.transform(df2['machine'])
    max_machine_idx = len(le_machine.classes_)

    machine_embedding = nn.Embedding(num_embeddings=max_machine_idx, embedding_dim=embedding_dim)
    return machine_embedding, df1, df2

def apply_embedding_and_scale_numerical(df, machine_embedding, embedding_dim):
    machine_indices = torch.tensor(df['machine_encoded'].values, dtype=torch.long)
    embedded = machine_embedding(machine_indices)

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('machine_encoded', errors='ignore')
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    numerical_features_tensor = torch.tensor(df[numerical_cols].values, dtype=torch.float)
    combined_features = torch.cat([embedded, numerical_features_tensor], dim=1)
    return combined_features, numerical_cols

def main():
    df1 = pd.read_csv('machine_charax.csv')
    df2 = pd.read_csv('Machine2.csv')
    embedding_dim = 1

    machine_embedding, df1, df2 = encode_and_embed(df1, df2, embedding_dim)

    combined_features_df1, numerical_cols_df1 = apply_embedding_and_scale_numerical(df1, machine_embedding, embedding_dim)
    combined_features_df2, numerical_cols_df2 = apply_embedding_and_scale_numerical(df2, machine_embedding, embedding_dim)

    # For df1
    combined_features_array_df1 = combined_features_df1.detach().numpy()
    column_names_df1 = ['embedded_machine_dim_{}'.format(i) for i in range(embedding_dim)] + list(numerical_cols_df1)
    combined_features_df1_final = pd.DataFrame(combined_features_array_df1, columns=column_names_df1)
    combined_features_df1_final.to_csv('combined_features_table1.csv', index=False)

    # For df2 - mirroring the operations done for df1
    combined_features_array_df2 = combined_features_df2.detach().numpy()
    column_names_df2 = ['embedded_machine_dim_{}'.format(i) for i in range(embedding_dim)] + list(numerical_cols_df2)
    combined_features_df2_final = pd.DataFrame(combined_features_array_df2, columns=column_names_df2)
    combined_features_df2_final.to_csv('combined_features_table2.csv', index=False)

if __name__ == "__main__":
    main()

