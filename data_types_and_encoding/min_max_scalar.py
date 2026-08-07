import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    'Temperature': [10.0, 20.0, 30.0],
    'Price': [100000.0, 300000.0, 500000.0]
}

df = pd.DataFrame(data)

print(df)

min_max_scalar = MinMaxScaler()
min_max_scaled = min_max_scalar.fit_transform(df)
df_minmax = pd.DataFrame(min_max_scaled, columns=['Temp_MinMax', 'Price_MinMax'])

print(df_minmax)

df_final = pd.concat([df, df_minmax], axis = 1)

print(df_final)