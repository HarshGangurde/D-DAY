## pip install tensorflow
import tensorflow as tf
a = tf.constant([2, 3, 4])         
b = tf.constant([5, 6, 7])          
print("Tensor a:", a)
print("Tensor b:", b)
add_result = tf.add(a, b)         
mul_result = tf.multiply(a, b)      
print("\nAddition:", add_result)
print("Multiplication:", mul_result)
c = a + b
print("\nEager Execution Example (a + b):", c)


import pandas as pd
import numpy as np
import seaborn as sns
df = sns.load_dataset('iris')
print("Original Iris Dataset (first 5 rows):")
print(df.head())
df.loc[2, 'sepal_length'] = np.nan
df.loc[5, 'petal_width'] = np.nan
df['sepal_length'].fillna(df['sepal_length'].mean(), inplace=True)
df['petal_width'].fillna(df['petal_width'].median(), inplace=True)
print("\nAfter Filling Missing Data:")
print(df.head())
num_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for col in num_cols:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
print("\nAfter Normalizing Numerical Data:")
print(df.head())
df['species'] = df['species'].map({
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
})
print("\nAfter Encoding Categorical Data:")
print(df.head())


## pip install seaborn matplotlib pandas
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
iris = sns.load_dataset("iris")
print("First 5 rows of the dataset:")
print(iris.head())
iris.hist(figsize=(8, 6), color='skyblue', edgecolor='black')
plt.suptitle("Histogram of Iris Features", fontsize=14)
plt.show()
plt.figure(figsize=(5, 4))
plt.scatter(iris['petal_length'], iris['petal_width'], color='purple')
plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()
plt.figure(figsize=(5, 4))
sns.heatmap(iris.drop('species', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Iris Features")
plt.show()
sns.pairplot(iris, hue="species")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()


#pip install gensim
from gensim.models import Word2Vec
sentences = [
    ["artificial", "intelligence", "is", "cool"],
    ["machine", "learning", "is", "fun"],
    ["ai", "learning", "uses", "neural", "networks"]]
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, sg=1)
print("Vector for 'learning':", model.wv['learning'])
print("Most similar to 'learning':", model.wv.most_similar('learning'))















