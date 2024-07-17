import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

data = pd.read_csv('data.csv')
data['YEAR'] = pd.to_datetime(data['YEAR'], format='%Y')

data.set_index('YEAR', inplace=True)

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

z_score = 1


def draw_z_score(x, cond, mu=0, sig=1):
    y = stats.norm.pdf(x, mu, sig)
    z = x[cond]
    plt.plot(x, y)
    plt.fill_between(z, 0, stats.norm.pdf(z, mu, sig), alpha=.2)
    plt.grid()
    plt.show()


x = np.arange(-3, 3, 0.001)
draw_z_score(x, x < z_score)
