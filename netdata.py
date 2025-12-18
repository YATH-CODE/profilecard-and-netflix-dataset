import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r'C:\Users\soniy\.vscode\assignment1(gdg)ai ml\netflix_output.csv')
print(df.head(10))
print('\n')
print(df.shape)
print('\n')
df.info()
print('\n')
print(df.describe())
print("Missing values in columns: ")
print(df.isnull().sum())

movie_mean = df[df['type'] == 'Movie']['duration_minutes'].mean()
tv_mean = df[df['type'] == 'TV Show']['seasons'].mean()

print("Mean Movie Duration:", movie_mean)
print("Mean TV Show Duration(season):", tv_mean)

print("Duplicates: ")
print(df.duplicated(subset='show_id').sum())
df = df.drop_duplicates(subset='show_id')
print("After removing duplicates shape will be: ")
print(df.shape)
df['country']=df['country'].fillna("Unknown")
df['director']=df['director'].fillna("Not listed the director")
print("Filled the country and director that were not placed and now are placed with Unknown and Not listed the director respectively for the above")

print(df[['country','director']].head(10))

df['duration_minutes'] = np.where( df['type'] == 'Movie', df['duration'].str.split().str[0].astype(float), np.nan )
df['seasons'] = np.where( df['type'] == 'TV Show', df['duration'].str.split().str[0].astype(float), np.nan )
print("Extracting new columns for movies with more than 90 minutes and tv show with more than 2 season")
print(df[['type', 'duration', 'duration_minutes', 'seasons']].head(10))

df['Is_recent']=np.where(df['release_year']>=2015,1,0)
print("Is_recent=1 then release year is greater than or equal to 2015 created a binary")
print(df[['Is_recent','release_year']].head(10))

df['type'].value_counts().plot(kind='bar')
plt.show()

plt.hist(df['release_year'])
plt.show()

top_countries = df['country'].value_counts().head(10)

top_countries.plot(kind='bar')
plt.title("Top 10 Countries by Number of Releases")
plt.xlabel("Country")
plt.ylabel("Number of Releases")
plt.show()

movies = df[df['type'] == 'Movie']


sns.boxplot(x='Is_Recent', y='duration_minutes', data=movies)
plt.title("Movie Duration: Recent vs Older")
plt.xlabel("Is_Recent (1 = Recent, 0 = Older)")
plt.ylabel("Duration (minutes)")
plt.show()

num_df = df[['release_year', 'duration_minutes', 'seasons', 'Is_Recent']]
corr = num_df.corr()
sns.heatmap(corr, annot=True)
plt.show()