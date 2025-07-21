
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('student-mat.csv', sep=';')

# 1. EDA
print("==== EDA SINGKAT ====")
print(df.info())
print(df.describe())
print(df.head())

# Visualisasi distribusi G3
plt.figure(figsize=(6,4))
sns.histplot(df['G3'], bins=20, kde=True, color='royalblue')
plt.title('Distribusi Nilai Akhir (G3)')
plt.xlabel('Nilai Akhir (G3)')
plt.ylabel('Jumlah Siswa')
plt.tight_layout()
plt.savefig('dist_g3.png')
plt.close()

# Heatmap korelasi
plt.figure(figsize=(8,6))
sns.heatmap(df[['G1','G2','G3','studytime','absences','failures']].corr(), annot=True, cmap='YlGnBu')
plt.title('Korelasi Variabel Penting')
plt.tight_layout()
plt.savefig('heatmap_corr.png')
plt.close()

# 2. REGRESI LINEAR
from sklearn.linear_model import LinearRegression

# Studytime vs G3
X1 = df[['studytime']]
y = df['G3']
lr1 = LinearRegression().fit(X1, y)
print(f'Regresi Studytime-G3, Koefisien: {lr1.coef_[0]:.2f}, Intercept: {lr1.intercept_:.2f}')

plt.figure(figsize=(6,4))
sns.regplot(x='studytime', y='G3', data=df, ci=None, scatter_kws={'alpha':0.5})
plt.title('Regresi: Studytime vs G3')
plt.tight_layout()
plt.savefig('reg_studytime_g3.png')
plt.close()

# Absences vs G3
X2 = df[['absences']]
lr2 = LinearRegression().fit(X2, y)
print(f'Regresi Absences-G3, Koefisien: {lr2.coef_[0]:.2f}, Intercept: {lr2.intercept_:.2f}')

plt.figure(figsize=(6,4))
sns.regplot(x='absences', y='G3', data=df, ci=None, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Regresi: Absences vs G3')
plt.tight_layout()
plt.savefig('reg_absences_g3.png')
plt.close()

# 3. CLUSTERING Berdasarkan Absensi
from sklearn.cluster import KMeans

X_clust = df[['absences']]
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_clust)
df['cluster_absen'] = kmeans.labels_

plt.figure(figsize=(7,5))
sns.scatterplot(x='absences', y='G3', hue='cluster_absen', palette='Set1', data=df, alpha=0.7)
plt.title('Cluster Berdasarkan Absensi')
plt.tight_layout()
plt.savefig('cluster_absensi.png')
plt.close()

# 4. KLASIFIKASI: Prediksi Lulus/Gagal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df['lulus'] = (df['G3'] >= 10).astype(int)
fitur = ['studytime', 'failures', 'absences']
X = df[fitur]
y = df['lulus']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

importances = clf.feature_importances_
plt.figure(figsize=(6,4))
sns.barplot(x=fitur, y=importances)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nSelesai! Semua plot sudah tersimpan ke file PNG.")
