import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

data = pd.read_csv('data_cleaned.csv')

data['YEAR'] = pd.to_datetime(data['YEAR'], format='%Y')

data.set_index('YEAR', inplace=True)
data_2017 = data[data.index.year == 2022]

data_2017.plot(x='REGION', y='SALARY', kind='bar', color='orange')
plt.xlabel("Province", fontsize=14)
plt.ylabel("Salary")
plt.title("Salary in 2022 by Province")
plt.subplots_adjust(bottom=0.25)
plt.tight_layout()

plt.grid(True)
plt.show()
mean_salary_per_salary = data.groupby('YEAR')['SALARY'].mean()

plt.plot(mean_salary_per_salary, marker='o', label='Mean Salary')
plt.title('Average Salary in Indonesia by Year (1997-2022)')
plt.xlabel('Year')
plt.ylabel('Mean Salary')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=mean_salary_per_salary.index.year, y=mean_salary_per_salary.values, palette="rocket")
plt.title('Average Salary in Indonesia by Year (1997-2022)')
plt.xlabel('Year')
plt.ylabel('Average Salary')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.show()

salary_2022 = data.loc[data.index.year == 2022, 'SALARY'].values
mean_salary_2022 = np.mean(salary_2022)
median_2022 = np.median(salary_2022)
mode_2022 = stats.mode(salary_2022)
std_2022 = np.std(salary_2022)
z_scores = (salary_2022 - mean_salary_2022) / std_2022
print("Z-Scores for Salary in 2017:")
for i, z in enumerate(z_scores):
    print(f"Salary {i + 1}: {z:.5f}")


def draw_z_score(x, cond, mu=0, sig=1):
    y = stats.norm.pdf(x, mu, sig)
    z = x[cond]
    plt.plot(x, y)
    plt.fill_between(z, 0, stats.norm.pdf(z, mu, sig), alpha=.2)
    plt.grid()
    plt.show()


x = np.arange(-3, 3, 0.001)
draw_z_score(x, x < 0.95110)

plt.figure(figsize=(12, 6))
sns.kdeplot(salary_2022, shade=True, color='orange',fill=True)

plt.xlabel("Province", fontsize=14)
plt.ylabel("Salary", fontsize=14)
plt.title("Salary in 2022 by Province")

plt.show()

quartile_25 = np.percentile(salary_2022, 25)
quartile_75 = np.percentile(salary_2022, 75)
quartile_50 = np.percentile(salary_2022, 50)

print(f"quartile_25: {quartile_25}")
print(f"quartile_75: {quartile_75}")
print(f"quartile_50: {quartile_50}")