import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter

data_dir = '../../data/raw'
os.makedirs(data_dir, exist_ok=True)

csv_path = os.path.join(data_dir, 'keyword_search_results.csv')

try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"File not found: {csv_path}")
    exit()

print(data.head())
print(data.info())

# SOC Code Coverage
soc_code_coverage = data.groupby('Keyword')['SOC Code'].nunique().reset_index()
soc_code_coverage.columns = ['Keyword', 'Unique SOC Codes']
print("SOC Code Coverage:")
print(soc_code_coverage)

# SOC Code Distribution by Keyword
plt.figure(figsize=(14, 10))
sns.countplot(y='SOC Code', hue='Keyword', data=data, order=data['SOC Code'].value_counts().index, palette='viridis')
plt.title('SOC Code Distribution by Keyword')
plt.xlabel('Count')
plt.ylabel('SOC Code')
plt.legend(title='Keyword', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('../../results/soc_code_distribution_by_keyword.png')
plt.show()

# Keyword-Specific Occupation Distribution
occupation_distribution = data.groupby(['Keyword', 'Job Title']).size().reset_index(name='Counts')
occupation_distribution_pivot = occupation_distribution.pivot(index='Job Title', columns='Keyword', values='Counts').fillna(0)
plt.figure(figsize=(14, 10))
sns.heatmap(occupation_distribution_pivot, cmap="YlGnBu")
plt.title('Occupation Distribution by Keyword')
plt.xlabel('Keyword')
plt.ylabel('Job Title')
plt.savefig('../../results/occupation_distribution_by_keyword.png')
plt.show()

# SOC Code Distribution Visualization
plt.figure(figsize=(14, 10))
sns.countplot(y='SOC Code', data=data, order=data['SOC Code'].value_counts().index, palette='viridis')
plt.title('SOC Code Distribution')
plt.xlabel('Count')
plt.ylabel('SOC Code')
plt.savefig('../../results/soc_code_distribution.png')
plt.show()

# Clustering Analysis
vectorizer = TfidfVectorizer().fit_transform(data['Job Title'])
vectors = vectorizer.toarray()
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(vectors)
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(reduced_data)

plt.figure(figsize=(12, 8))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('Keyword Clustering based on Occupation Titles')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig('../../results/keyword_clustering.png')
plt.show()

# Save the findings to CSV files
soc_code_coverage.to_csv('../../data/processed/soc_code_coverage.csv', index=False)
occupation_distribution.to_csv('../../data/processed/occupation_distribution.csv', index=False)

# Save the findings to a text file
with open('../../results/keyword_search_evaluation_summary.txt', 'w') as f:
    f.write("Keyword Search Evaluation Summary\n")
    f.write("===============================\n\n")
    f.write("SOC Code Coverage:\n")
    f.write(soc_code_coverage.to_string(index=False))
    f.write("\n\nClustering Analysis:\n")
    f.write("PCA and KMeans clustering were performed on the occupation titles based on keywords.\n")
    f.write("The results are visualized in the keyword_clustering.png file.\n")

print("Enhanced EDA completed. Findings saved to CSV and text files.")