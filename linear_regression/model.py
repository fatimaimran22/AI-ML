import pandas as pd
import numpy as np

df = pd.read_csv("linear_regression/train.csv")

print(f"--------Size of Dataset:----------\nHouses: {df.shape[0]}\nFeatures: {df.shape[1]}")  #(rows, columns)

# print(f"\nDataset:\n{df.head(2)}")

# print(df.info())

"""

int64    → integer numbers
float64  → decimal numbers
object   → usually text/categorical data
bool     → True/False

"""

# print(df.columns)
print(f"\n-----------SALESPRICE--------------")
print(df["SalePrice"].describe())

"""
max > 75% 
The maximum is much larger than the typical values → possible outlier.

Mean > median suggests the distribution may be right-skewed.

Spread:

mean = 180,000
std  = 79,000

→ Prices vary quite a lot.

"""

print("\n-----------Missing Values-----------")
# print(df.isnull().sum())
print(df.isnull().sum().sort_values(ascending = False).head(20))    
"""
Gives 20 columns with the most missing values
ascending=False means largest → smallest.
.head(20)-->Takes only the first 20 rows of missing columns output.

"""