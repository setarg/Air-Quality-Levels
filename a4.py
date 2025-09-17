import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Read data CSV file
df = pd.read_csv(r'C:\\Users\\tungs\\AirQualityUCI.csv', delimiter=';')
#Last 2 columns are unnamed with NaN values, drop them
df = df.iloc[:, :-2]
df = df.dropna()
#Hour feature
df['Hour'] = df['Time'].str.split('.').str[0].astype(int)

#for PCA processing, ignore Date/Time as they are not numeric
dfpca = df.iloc[:, 2:]

#Replace comma as delimiter for certain features with period for numeric processing
dfpca = dfpca.applymap(lambda x: str(x).replace(',','.'))
print(dfpca.head())

#PCA analysis
pca5 = PCA(n_components=5)
X = dfpca.values
rdf = pca5.fit_transform(X)

#Consolidate data
rdf = pd.DataFrame(rdf, columns= ['PC1','PC2','PC3','PC4','PC5'])
rdf = pd.concat([df['Hour'].reset_index(drop=True), rdf.reset_index(drop=True)], axis = 1)

print(rdf)

#K-Means clustering and plot
kmeans = KMeans(n_clusters=4) 
rdf['Cluster'] = kmeans.fit_predict(rdf.iloc[:, 1:])

plt.scatter(rdf['Hour'], rdf['PC1'], c=rdf['Cluster'], s=10, alpha=0.8)
plt.xlabel('Hour of the Day')
plt.ylabel('Pollution')
plt.title('Hour vs Pollution')
plt.show()