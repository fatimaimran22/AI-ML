import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

data = {
    "city": ["Lahore", "London", "Karachi", "Lahore"],
    "size": ["small", "medium", "large", "medium"]
}

df = pd.DataFrame(data)

# One-Hot Encoding
city_encoded = pd.get_dummies(df["city"], prefix="city")
print(city_encoded)

# Ordinal Encoding
order = [["small", "medium", "large"]]
ordinal_encoder = OrdinalEncoder(categories=order)
size_encoded = pd.DataFrame(ordinal_encoder.fit_transform(df[["size"]]), columns=["size_encoded"])
print(size_encoded)

# Label Encoding (target variable)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(["cat", "dog", "cat"])
print(labels)


# Combine everything
df_final = pd.concat([df, city_encoded, size_encoded], axis=1)

print(df_final)