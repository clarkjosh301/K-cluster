import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
cars_csv = r'C:\Users\hcpl_bel\Downloads\carsdata\cars.csv'
cars_df = pd.read_csv(cars_csv)
cars_df.replace(' ', np.nan, inplace=True)
cars_df.dropna(axis=0, how="any", inplace=True)
cars_df[' cubicinches'] = pd.to_numeric(cars_df[' cubicinches'])
cars_df[' weightlbs'] = pd.to_numeric(cars_df[' weightlbs'])
cars_df[' cubicinches'] = cars_df[' cubicinches'].astype(int)
cars_df[' weightlbs'] = cars_df[' weightlbs'].astype(int)
cleaned_brand = {' brand': {' US.': 0, ' Europe.': 1, ' Japan.': 2}}
cars_df.replace(cleaned_brand, inplace=True)
X = cars_df.drop([' brand'], axis=1)
y = cars_df[' brand']
kmc=KMeans(n_clusters=3)
KMmodel = kmc.fit(X)
pd.crosstab(y,KMmodel.labels_)
#model isn't very accurace, decide to test correlation 
import seaborn as sns
import matplotlib.pyplot as plt
corr = cars_df[cars_df.columns].corr()
sns.heatmap(corr, cmap='YlGnBu', annot = True)
#drop variables with <50% correlation
X = cars_df.drop([' brand', ' cylinders', ' year', ' hp', ' time-to-60'], axis=1)
KMmodel = kmc.fit(X)
cars_df['prediction']=KMmodel.labels_
f, (ax1) = plt.subplots(1, sharey=True, figsize=(12,6))
ax1.set_title('K Means')
ax1.scatter(cars_df[' weightlbs'],cars_df[' cubicinches'],cars_df['mpg'],c=cars_df['prediction'],cmap='rainbow')
plt.show()

