import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn

def load_data(file_path1, file_path2):
    df = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)
    return df, df2

def encode_machines(df, df2):
    all_machines = pd.concat([df['machine'], df2['machine']]).unique()
    le_machine = LabelEncoder().fit(all_machines)
    df['machine_encoded'] = le_machine.transform(df['machine'])
    df2['machine_encoded'] = le_machine.transform(df2['machine'])
    return df, df2, le_machine

def embed_and_scale(df, df2, le_machine):
    embedding_dim = 1  # Example embedding dimension
    max_machine_idx = len(le_machine.classes_)
    machine_embedding = nn.Embedding(num_embeddings=max_machine_idx, embedding_dim=embedding_dim)
    
    # Embedding
    embedded_df = machine_embedding(torch.tensor(df['machine_encoded'].values, dtype=torch.long))
    embedded_df2 = machine_embedding(torch.tensor(df2['machine_encoded'].values, dtype=torch.long))
    
    # Scaling
    scaler = StandardScaler()
    numerical_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns if col != 'machine_encoded']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    numerical_cols2 = [col for col in df2.select_dtypes(include=['float64', 'int64']).columns if col != 'machine_encoded']
    df2[numerical_cols2] = scaler.fit_transform(df2[numerical_cols2])
    
    return df, df2, embedded_df, embedded_df2, numerical_cols, numerical_cols2

def combine_features(df, df2, embedded_df, embedded_df2, numerical_cols, numerical_cols2, embedding_dim):
    combined_features_df = torch.cat([embedded_df, torch.tensor(df[numerical_cols].values, dtype=torch.float)], dim=1)
    combined_features_df2 = torch.cat([embedded_df2, torch.tensor(df2[numerical_cols2].values, dtype=torch.float)], dim=1)
    
    column_names = ['embedded_machine_dim_{}'.format(i) for i in range(embedding_dim)] + list(numerical_cols)
    combined_features_array = combined_features_df.detach().numpy()
    combined_features_df_final = pd.DataFrame(combined_features_array, columns=column_names)
    
    return combined_features_df_final

def main():
    df, df2 = load_data('machine_charax.csv', 'Machine2.csv')
    df, df2, le_machine = encode_machines(df, df2)
    df, df2, embedded_df, embedded_df2, numerical_cols, numerical_cols = embed_and_scale(df, df2, le_machine)
    combined_features_df = combine_features(df, df2, embedded_df, embedded_df2, numerical_cols, numerical_cols, 1)
    print(combined_features_df.head())
    combined_features_df.to_csv('combined_features_table.csv', index=False)

if __name__ == '__main__':
    main()

