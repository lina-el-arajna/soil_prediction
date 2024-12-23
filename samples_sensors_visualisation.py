import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/sensor_data.csv')
df.columns = df.columns.str.strip()
print("Columns in the DataFrame:", df.columns)

print(df.head())

# Remove the 'Sample ID' column
df = df.drop(columns=['Sample ID'])

# Scatter plot of Temperature vs Moisture with pH levels and color by Quality
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Tempreture', y='Moisture', hue='ph', palette='coolwarm', size='ph', sizes=(20, 200), style='Quality')
plt.title('Temperature vs Moisture with pH levels')
plt.xlabel('Temperature')
plt.ylabel('Moisture')
plt.legend(title='pH and Quality', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Pairplot of variables colored by Moisture
sns.pairplot(df, hue='Moisture', palette='coolwarm')
plt.suptitle('Pairplot of Variables', y=1.02)
plt.show()

# Distribution plot of pH levels
plt.figure(figsize=(12, 6))
sns.histplot(df['ph'], kde=True, bins=10, color='blue')
plt.title('Distribution of pH levels')
plt.xlabel('pH')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of Temperature vs pH colored by Moisture levels and style by Quality
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Tempreture', y='ph', hue='Moisture', palette='coolwarm', style='Quality')
plt.title('Temperature vs pH colored by Moisture levels')
plt.xlabel('Temperature')
plt.ylabel('pH')
plt.legend(title='Moisture and Quality', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 6))
numeric_df = df[['Tempreture', 'Moisture', 'ph']]
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Correlation Matrix')
plt.show()
