import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

print("Data Head:\n", df.head())

categorical_vars = ['Surname', 'Geography', 'Gender']

for col in categorical_vars:
    df[col], _ = pd.factorize(df[col])

df = df.apply(pd.to_numeric, errors='coerce')

df.dropna(inplace=True)

corr_matrix = df.corr()

corr_with_exited = corr_matrix['Exited'].sort_values(ascending=False)

print("\nCorrelations with 'Exited':\n", corr_with_exited)
