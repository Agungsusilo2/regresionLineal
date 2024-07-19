import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

data = pd.read_csv('data_cleaned.csv')

data["YEAR"] = pd.to_datetime(data["YEAR"], format="%Y")
data.set_index("YEAR", inplace=True)
mean_salary = data.groupby("YEAR")["SALARY"].mean()
mean_salary_df = mean_salary.reset_index()

plt.scatter(mean_salary_df['YEAR'].dt.year, mean_salary_df['SALARY'], color="orange")
plt.xlabel("Year")
plt.ylabel("Salary")
plt.xlim(1995, 2025)
plt.grid(True)
plt.show()

X = mean_salary_df['YEAR'].dt.year.values.reshape(-1, 1)
y = mean_salary_df['SALARY'].values
print(X)
print(y)


scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()


plt.scatter(mean_salary_df['YEAR'].dt.year, y_scaled, color='orange')
plt.xlabel("Year")
plt.ylabel("Scaled Salary")
plt.xlim(1995, 2025)
plt.grid(True)
plt.show()


model = LinearRegression()
model.fit(X, y_scaled)

X_vis = np.array([1995, 2025]).reshape(-1, 1)
y_vis = model.predict(X_vis)

plt.scatter(mean_salary_df["YEAR"].dt.year, y_scaled, color='orange')

plt.plot([1995, 2025], y_vis, '-r')
plt.xlabel("Year")
plt.ylabel("Salary")
plt.xlim(1995, 2025)
plt.grid(True)
plt.show()

print(f'intercept: {model.intercept_}')
print(f'slope: {model.coef_}')

print(f'X:\n{X}\n')
print(f'X : flatter :{X.flatten()}\n')
print(f'Y:\n{y}\n')

var_x = np.var(X.flatten(), ddof=1)
print(f'var_x: {var_x}')

cov_xy = np.cov(X.flatten(), y, ddof=1)[0, 1]
print(f'cov_xy: {cov_xy}')

slope = cov_xy / var_x
print(f'slope: {slope}')

intercept = np.mean(y) - slope * np.mean(X)
print(f'intercept: {intercept}')


year = np.arange(2050, 2054).reshape(-1, 1)
prediction = model.predict(year)

prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()

for year, prediction in zip(year, prediction):
    print(f'{year}: {prediction}')

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y_scaled,
                                                    test_size=0.4,
                                                    random_state=1)

print(f'X_train: {X_train}')
print(f'X_test: {X_test}')
print(f'y_train: {y_train}')
print(f'y_test: {y_test}')

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'r2: {r2}')

ss_res = sum([(y_i - model.predict(x_i.reshape(-1, 1))[0]) ** 2
              for x_i, y_i in zip(X_test, y_test)])

print(f'ss_res: {ss_res}')

mean_y = np.mean(y_test)
ss_tot = sum([(y_i - mean_y) ** 2 for y_i in y_test])

print(f'ss_tot: {ss_tot}')

r_squared = 1 - (ss_res / ss_tot)
print(f'R2: {r_squared}')

n = X.shape[0]
k = X.shape[1]+1
adj_r2 = 1 - (1 - r2) * (n-1)/(n-k-1)
print(f'adj_r2: {adj_r2}')

corr, p_value = pearsonr(mean_salary_df['YEAR'].dt.year, mean_salary_df['SALARY'])
print(f"Pearson correlation coefficient: {corr}")
print(f"P-value: {p_value}")

alpha = 0.05
if p_value < alpha:
    print("Tolak hipotesis nol - ada korelasi yang signifikan.")
else:
    print("Gagal menolak hipotesis nol - tidak ada korelasi yang signifikan.")