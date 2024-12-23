import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/linae/Documents/LinkSquare/Export/s1.csv', delimiter=';', header=None, usecols=range(1, 550))

df = df.replace(',', '.', regex=True)

X = df.iloc[0].astype(float)
Y1 = df.iloc[1].astype(float)
Y2 = df.iloc[2].astype(float)

plt.figure(figsize=(10, 6))

plt.plot(X, Y1, label='Y1 (Intensity)', marker='o')
plt.plot(X, Y2, label='Y2 (Intensity)', marker='o')

plt.title('Intensity vs Wavelength')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (A.U.)')

plt.legend()

plt.show()

df_stats = pd.DataFrame({
    'X': X,
    'Y1': Y1,
    'Y2': Y2
})
statistics = df_stats.describe()
print(statistics)
