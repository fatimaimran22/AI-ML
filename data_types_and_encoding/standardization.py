import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    'Temperature': [10.0, 20.0, 30.0],
    'Price': [100000.0, 300000.0, 500000.0]
}

df = pd.DataFrame(data)

print(df)

std_scalar = StandardScaler()
std_scaled = std_scalar.fit_transform(df)
df_std = pd.DataFrame(std_scaled, columns=["Temp_std", "Price_std"])

print(df_std)

df_final = pd.concat([df, df_std], axis= 1)

print(df_final)