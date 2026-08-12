import pandas as pd

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


print("\n-----------Column Data Types Total----------")
print(df.dtypes.value_counts())

print("\n-----------Numerical Columns----------")
print(df.select_dtypes(include="number").columns)


print("\n-----------Categorical/Text Columns----------")
print(df.select_dtypes(include="object").columns)



import matplotlib.pyplot as plt

plt.hist(df["SalePrice"], bins=30)
plt.xlabel("Sales Price")
plt.ylabel("Number of Houses")
plt.title("Distribution of House Prices")
# plt.show()


plt.scatter(df["GrLivArea"], df["SalePrice"])   #GrLivArea (above-ground living area).
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.title("Living Area vs Sale Price")
# plt.show()


print("\n-----------Finding Outliers----------")

Q1 = df["SalePrice"].quantile(0.25)
Q3 = df["SalePrice"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["SalePrice"] < lower) | (df["SalePrice"] > upper)
]

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower)
print("Upper bound:", upper)
print("Number of outliers:", len(outliers))

print("\n-----------Outliers----------")
print(outliers["SalePrice"].min())
print(outliers["SalePrice"].max())

print(outliers[["GrLivArea", "OverallQual", "SalePrice"]].sort_values("SalePrice").tail(10))


"""
IQR identified 61 potential outliers in SalePrice. 
Inspection of the extreme observations showed that they correspond to 
legitimate high-value properties rather than obvious data-entry errors. 
Therefore, they were retained rather than removed.

"""

import seaborn as sns

numeric_df = df.select_dtypes(include="number")

corr = numeric_df.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
# plt.show()


"""
|      VIF | Interpretation           |
| -------: | ------------------------ |
|   1      | No multicollinearity     |
|  1 - 5   | Usually okay             |
|  5 - 10  | Concerning               |
|   > 10   | Strong multicollinearity |


"""

print("\n-----------VIF----------")

from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df.select_dtypes(include= "number").drop(columns = ["SalePrice"])
X = X.fillna(X.median())

vif = pd.DataFrame()

vif["Feature"] = X.columns
vif["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print(vif.sort_values("VIF", ascending=False).head(20))


print("\n-----------Train Test----------")
X = df.drop(columns= ["SalePrice"])
y = df["SalePrice"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

num_cols = X_train.select_dtypes(include="number").columns
cat_cols = X_train.select_dtypes(include="object").columns

print("Numerical:", len(num_cols))
print("Categorical:", len(cat_cols))


#print("\n-----------Fill Missing Values and Encoding---------")

from sklearn.compose import ColumnTransformer   #ColumnTransformer lets you apply different preprocessing to different columns.
from sklearn.pipeline import Pipeline   #Pipeline lets you chain multiple preprocessing steps together.
from sklearn.impute import SimpleImputer    #SimpleImputer handles missing values (NaN).
from sklearn.preprocessing import OneHotEncoder, StandardScaler #OneHotEncoder converts categorical values into numbers.
from sklearn.linear_model import LinearRegression


"""
ColumnTransformer says:

num_cols
   ↓
numeric_transformer
   ↓
median imputation


cat_cols
   ↓
categorical_transformer
   ↓
most-frequent imputation
   ↓
one-hot encoding


                 DATA
                  │
          ┌───────┴────────┐
          ↓                ↓
     Numerical         Categorical
      columns             columns
          │                │
       median          most frequent
      imputation         imputation
          │                │
          │            one-hot encoding
          │                │
          └───────┬────────┘
                  ↓
           PREPROCESSED DATA

"""


numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scalar", StandardScaler())
])

categorical_imputer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoding", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_imputer, cat_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


model.fit(X_train, y_train) #   = train the model.

"""
fit() = learn

transform() = apply what was learned

fit_transform() = learn + apply

"""

print("\n-----------Prediction---------")

y_pred = model.predict(X_test)

# print(y_pred[:10])


comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(comparison.head(10))


print("\n-----------Evaluate Assumptions---------")

residuals = y_test - y_pred
# print(residuals.head())

print("Mean residual:", residuals.mean())


print("\n-----------1) Linearity: Residual-Fitted PLot---------")

import matplotlib.pyplot as plt

plt.scatter(y_pred, residuals)
plt.axhline(y=0, linestyle="--")

plt.xlabel("Fitted (Predicted) Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")

# plt.show()


print("\n-----------2) Independence: Durbin Watson Test---------")

from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(residuals)

print("Durbin-Watson Statistic:", dw)


"""

0        2        4
│────────│────────│
strong   no       strong
positive correlation  negative
correlation           correlation


"""

print("\n-----------3) Homoscedasticity: Breusch-Pagan test.---------") #Does the spread of residuals stay roughly constant across all predicted values?

from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm

X_bp = sm.add_constant(y_pred)

bp_test = het_breuschpagan(residuals, X_bp)

# print("LM Statistic:", bp_test[0])
print("LM p-value:", bp_test[1])
# print("F Statistic:", bp_test[2])
# print("F p-value:", bp_test[3])

"""
p-value > 0.05 →  no strong evidence of heteroscedasticity
p-value < 0.05 →  evidence of heteroscedasticity

Homoscedasticity is not satisfied. There is strong statistical evidence of heteroscedasticity.

This also matches with earlier: the residual plot had a cone/fan-like spread.

"""

if bp_test[1] < 0.05:
    print("Violated.")


print("\n-----------4) Normality of residuals: Q-Q plot and Shapiro-Wilk test.---------")

print("\n-----------Q-Q plot--------")
import scipy.stats as stats
import matplotlib.pyplot as plt

stats.probplot(residuals, dist="norm", plot=plt)

plt.title("Q-Q Plot of Residuals")
# plt.show()

print("\n-----------Shapiro-Wilk test---------")
stat, p = stats.shapiro(residuals)

print("Shapiro-Wilk statistic:", stat)
print("p-value:", p)

if p < 0.05:
    print("Violated.")

"""
p > 0.05 →  no strong evidence against normality
p < 0.05 → evidence that residuals are not normal

"""

print("\n-----------5) Multicollinearity: VIF---------")
"""
VIF ≈ 1       → no multicollinearity
VIF 1-5       → usually okay
VIF 5-10      → concerning
VIF > 10      → strong multicollinearity

"""

print(vif.sort_values("VIF", ascending=False).head(20))


print("\n-----------Evaluate Model---------")

y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

train_mae = mean_absolute_error(y_train, y_train_pred)  
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)

print("\n-----TRAINING------")
print("MAE:", train_mae)
print("MSE:", train_mse)
print("RMSE:", train_rmse)
print("R²:", train_r2)


test_mae = mean_absolute_error(y_test, y_pred)  #On average, how many dollars are we wrong by?
test_mse = mean_squared_error(y_test, y_pred)   #Like MAE, but large errors are punished much more heavily.
test_rmse = np.sqrt(test_mse)   
test_r2 = r2_score(y_test, y_pred)  #How much of the variation in house prices does the model explain?

print("\n------TEST--------")
print("MAE:", test_mae)
print("MSE:", test_mse)
print("RMSE:", test_rmse)
print("R²:", test_r2)

def mape(y_true, y_pred):
    return np.mean(
        np.abs((y_true - y_pred) / y_true)
    ) * 100

print("\n------Mean Absolute Percentage Error (MAPE)--------")

train_mape = mape(y_train, y_train_pred)
test_mape = mape(y_test, y_pred)

print("Train MAPE:", train_mape, "%")
print("Test MAPE:", test_mape, "%")

print("\n------Adjusted R²--------")

n_train = len(y_train)
n_test = len(y_test)

p = model.named_steps["preprocessor"].transform(X_train).shape[1]

train_adj_r2 = 1 - (1 - train_r2) * (n_train - 1) / (n_train - p - 1)

test_adj_r2 = 1 - (1 - test_r2) * (n_test - 1) / (n_test - p - 1)

print("Train Adjusted R²:", train_adj_r2)
print("Test Adjusted R²:", test_adj_r2)

"""
The linear regression model achieved an R² of 0.887 on the test set,
 explaining approximately 88.7% of the variance in house prices. 
 The test MAE was approximately $18,285 and the RMSE was approximately $29,476. 
 The test MAPE was 11.28%, indicating that predictions were on average around 11% away from actual prices. 
 The training R² (0.936) was higher than the test R² (0.887), but the relatively small gap of 0.049 suggests that severe overfitting is not present.

"""

features_names = model.named_steps["preprocessor"].get_feature_names_out()
coefficients = model.named_steps["regressor"].coef_

coef_df = pd.DataFrame({
    "Feature": features_names,
    "Coefficient": coefficients
})

coef_df["Absolute"] = coef_df["Coefficient"].abs()

# print(coef_df.sort_values("Absolute", ascending=False).head(20))


print("\n--- Numerical coefficients ---")

print(
    coef_df[
        coef_df["Feature"].str.startswith("num__")
    ].sort_values("Absolute", ascending=False).head(15)
)


print("\n--- Categorical coefficients ---")

print(
    coef_df[
        coef_df["Feature"].str.startswith("cat__")
    ].sort_values("Absolute", ascending=False).head(15)
)