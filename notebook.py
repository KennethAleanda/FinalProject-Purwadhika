# %% [markdown]
# # Final Project - 

# %% [markdown]
# ## Domain Knowledge
# 
# ### Context
# syarah.com adalah online marketplace untuk penjualan mobil bekas di saudi arabia, untuk menaikkan revenue akan menawarkan feature price suggestion kepada seller. feature ini akan dapat diakses oleh seller melalui subscription. dengan feature ini, seller dapat memasukkan input berupa attribute/feature dari mobil yang ingin dijual, outputnya adalah price range yang direkomendasikan untuk mobil dengan attribute/feature tersebut. 
# 
# ### Stakeholder
# syarah.com, syarah company (salah satu marketplace online untuk penjualan mobil bekas di Saudi Arabia)
# 
# ### Busines Problem
# 1. **menentukan harga yang comptetive pada mobil bekas adalah hal yang lumayan sulit dilakukan**. tidak semua seller mengetahui hal-hal mengenai mobil yang ingin dijual, dan mencari informasi mengenai harga pasar dari mobil dengan feature-feature detail tertentu (e.g. mileage, kondisi kesehatan ban, kondisi exterior mobil, warna mobil tertentu, kondisi mesin, modifikasi mobil, keseringan service, tempat service dst.) sangatlah memakan waktu dan pengetahuan mengenai detail-detail tersebut. untuk kebanyakan orang yang ingin menjal mobil bekas mereka, waktu yang dibutuhkan untuk meneliti harga yang kompetitif dapat bervariasi, mulai dari  beberapa jam hinggaa beberapa hari. ini melibatkan pengecakan daftar harga online, kunjungan ke dealer, dan pertimbangan faktor-faktor seperti merk, model, tahun, jarak, tempuh, kondisi, dan permintaan.[^1] [^2] [^3]
# 
# 2. **mengidentifikasi faktor-faktor apa saja yang mempengaruhi harga mobil pada saat tertentu memerlukan banyak waktu jika dilakukan secara manual.** seperti halnya menentukan harga, mencari tahu faktor-faktor apa saja yang dapat mempengaruhi harga mobil sangatlah memakan waktu. dan tidak semua seller mempunya waktu ataupun kemauan untuk me-researchnya
# 
# ### Goals
# 1. **Accurately Predict the price of used cars** : membuat model dapat membantu penjualan mobil bekas dengan menetapkan range harga yang kompetitif dan akurat / 20-25% MAPE
# 2. **Understanding Key Factors** : mengidentifikasi faktor-faktor penting yang dapat mempengaruhi harga mobil bekas dijual
# 
# ### Business Questions
# 1. **berapa prediksi harga dari sebuah mobil bekas dengan feature tertentu?** 
# 2. **feature mana yang berpengaruh dalam menetapkan harga mobil bekas?**
# 3. **bagaimana price distribution dari mobil bekas?**
# 4. **apa saja popular brand dari setiap region categorynya?**
# 5. **apa saja type mobil yang populer dari setiap region categorynya?**
# 6. **gear and fuel type apa saja yang populer?**
# 7. **bagaimana relasi antara mileage dan age dari mobil bekas dengan popularitynya?**
# 
# ### Evaluation Metric
# - MedAE : Median Absolute Error adalah metric yang robust kepada outlier, loss dihitung dengan mengambil median dari semua absoulute differences antara target dan prediksi. MedAE dipilih karena data yang akan diprediksi mempunyai outlier yang cukup signifikan.    
# - MAE : Mean Absolute Error adalah metric yang digunakan untuk memprediksi range harga. range harga yaitu harga prediksi machine learning ditambang dan dikurangi MAE.
# - MAPE : Mean Absolute Percentage Error adalah bentuk persentase dari MAE, yang digunakan untuk merepresentasikan error rate pada prediksi machine learningnya
# 
# ### Project Limitation
# - dataset row yang memiliki value "True" pada column "negotiable" di drop karena memiliki nilai value 0 pada column "Price"
# - dataset berisi mobil bekas dari tahun 2003 - 2021
# - dataset berisi mobil bekas dengan ukuran engine size 1.0 - 8.0
# - dataset berisi mobil bekas dengan mileage 100 - 432000
# - dataset berisi mobil bekasi dengan harga 4000 - 1150000
# - dataset tidak memiliki population size untuk column regionnya, jadi diharuskan untuk mencari populasi untuk setiap valuenya dari external source, untuk men-categorisasikannya. 
# - Hardware Machine Learning
# 
# ### Data Features and Description
# nama dataset : Saudi Arabia Used Cars Dataset
# 
# data set berisi mobil bekas sebanyak 8035 records yang diambil dari syarah.com. setiap row merepresentasikan sebuah mobil bekas dengan informasi mengenai brand name, model, manufacturing year, origin, the color of the car, options, capacity of the engine, type of fuel, transmission type, the mileage that the car covered, region price, and negotiable
# 
# | Column | Data Type | Description |
# | --- | --- | --- |
# | Make | str - Nominal | nama brand dari mobil |
# | Type | str - Nominal | jenis dari mobil |
# | Year | int - Interval| tahun produksi mobil |
# | Origin | str - Nominal| asal mobil |
# | Color | str - Nominal| warna dari mobil |
# | Options | str - Ordinal| kelengkapan opsi yang ada pada mobil |
# | Engine_Size | float - Ratio| ukuran mesin yang digunakan oleh mobil |
# | Fuel_Type | str - Nominal | type bahan bakar yang digunakan oleh mobil |
# | Mileage | int - Ratio | jarak tempuh mobil |
# | Region | str - Nominal| wilayah tempat mobil dijual |
# | Price | int - Ratio| harga mobil |
# | Negotiable | bool - Bool| negosiasi harga mobil |
# 
# 
# 
# [^1]:https://www.kenresearch.com/industry-reports/singapore-use-car-market  
# [^2]:https://lotlinx.com/used-car-inventory-trends/   
# [^3]:https://markwideresearch.com/singapore-used-car-market/   

# %% [markdown]
# ## Data

# %%
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.formula.api import ols

from sklearn.cluster import KMeans

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from sklearn.metrics import  mean_absolute_error, median_absolute_error, mean_absolute_percentage_error, r2_score

from scipy.stats import randint, uniform

import pickle

# %%
# df = pd.read_csv("./Datasets/UsedCarsSA_Clean_EN.csv")

# %%
# df.head(10)

# %%
# df.shape

# %% [markdown]
# ## Explarotary Data Analysis (EDA)

# %% [markdown]
# Objective: To understand data distribution and condition for 
# preprocessing stage
# Elements:
# 1. Data Distribution Plot (Boxplot, QQplot, Histogram)
# 2. Data Correlation (Nominal and Ratio scale correlation)
# 3. Data Cardinalities (Count unique for categorical feature)
# 4. Identify missing value, outlier, anomaly, duplicates, etc
# 

# %% [markdown]
# ### Data Distribution Plot

# %%
# df.columns

# %%
# fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(12, 6))
# sns.boxplot(df['Price'], ax=ax_box)
# ax_box.set(title=f'{"Price"} box plot', xlabel='', ylabel='')    
# sns.histplot(df['Price'], ax=ax_hist, kde=True)
# ax_hist.set(title=f'{"Price"} hist plot', xlabel='', ylabel='')
# plt.tight_layout()
# plt.show()

# %% [markdown]
# Price tidak terdistribusi dengan normal  
# Price juga terlihat memiliki banyak outlier   
# price memiliki banyak yang bernilai 0 karena semua mobil yang memiliki nilai column negosiasi true maka price akan di set sebagai 0.

# %%
# listTemp = ['Year','Engine_Size', 'Mileage', 'Price']

# for i in listTemp:
#     fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(12, 6))
#     sns.boxplot(df[i], ax=ax_box)
#     ax_box.set(title=f'{i} box plot', xlabel='', ylabel='')    
#     sns.histplot(df[i], ax=ax_hist, kde=True)
#     ax_hist.set(title=f'{i} hist plot', xlabel='', ylabel='')
#     plt.tight_layout()
#     plt.show()

# %% [markdown]
# Year memiliki banyak outlier dan memiliki left skew   
# Engine_Size memiki outlier tetapi terlihat lebih merata dan memiliki right skew  
# Mileage memiliki banyak outlier dan memiliki right skew  
# Price memiliki banyak outlier dan memiliki right skew  

# %%
# listTemp = ['Make', 'Type', 'Origin', 'Color', 'Options', 'Fuel_Type', 'Gear_Type']

# for i in listTemp:
#     plt.figure(figsize=(6, df[i].nunique() * 0.3))

#     value_counts = df[i].value_counts().sort_values(ascending=False)
#     ax = sns.countplot(data=df, y=i, order=value_counts.index)
#     y_labels = [f"{category} ({count})" for category, count in value_counts.items()]
    
#     plt.title(f'Bar Plot for {i}', size=20, weight='bold')
#     plt.xlabel('Frequency')
#     plt.ylabel(i)
    
#     ax.set_yticks(range(len(y_labels))) 
#     ax.set_yticklabels(y_labels)
    
#     plt.show()

# %% [markdown]
# ### Data Correlation

# %%
# listTemp = ['Year','Engine_Size', 'Mileage']
# for i in listTemp:
#     df.plot(kind='scatter', y='Price', x=i)
#     plt.title(f'{i} dan Price')
#     plt.show()

# # %%
# plt.figure(figsize=(10,8))
# sns.heatmap(df.corr(method='spearman', numeric_only=True), annot=True)
# plt.title('Numerical Correlation')
# plt.show()

# %% [markdown]
# **Year** : 0.37, *postive medium* correlation kepada Price, semakin tinggi `year` (semakin baru) maka `Price` akan naik   
# **Engine_Size** : 0.096, *postive low* correlation kepada Price, semakin  besar `Engine_Size` maka `Price` akan naik sedikit   
# **Mileage** : -0.12, *negative low* correlation kepada Price, semakin besar `Mileage` maka `Price` akan turun sedikit   
# **Negotiable** : -0.82 *positive high* correlation kepada Price. pada kasus ini `Negotiable` berbanding terbalik terhadap `Price`, dimana jika `Negotiable` Postive (memiliki nilai / 1), maka `Price` akan 0, dan sebaliknya bila `Price` Postive (memiliki nilai) maka `Negotiable` akan 0     
#      
# ***Summary***
# - **Low Correlation** : `Engine_Size` dan `Mileage` memiliki low correlation kepada `Price`, yang berarti `Engine_Size` dan `Mileage` memiliki pengaruh yang kecil terhadap Price
# - **Medium Correlation** : `Year` memiliki medium correlation kepada `Price`, yang berarti nilai `Year` memiliki pengaruh yang cukup significant terhadap nilai `Price`
# - **High Correlation** : `Negotiable` memiliki high correlation kepada `Price`, yang berarti nilai `Negotiable` memilki pengaruh yang sangat significant terhadap nilai `Price`

# %%
# listTemp = ['Make','Type', 'Origin', 'Color', 'Options', 'Fuel_Type', 'Gear_Type', 'Region', 'Negotiable']

# for i in listTemp:
#     plt.figure(figsize=(10, df[i].nunique() * 0.3))

#     mean_prices = df.groupby(i)['Price'].mean().sort_values(ascending=False)
#     ax = sns.barplot(data=df, x='Price', y=i, order=mean_prices.index, errorbar=None)
#     y_labels = [f"{category} ({price:.2f})" for category, price in mean_prices.items()]

#     plt.title(f'{i} dan Price')

#     ax.set_yticks(range(len(y_labels)))
#     ax.set_yticklabels(y_labels)
#     plt.show()

#     model = ols(f'Price ~ C({i})', data=df).fit()
#     anova_table = sm.stats.anova_lm(model, typ=2)
    
#     print(f'ANOVA table for {i} and Price:')
#     print(anova_table)
#     print('\n')

# %% [markdown]
# ***Columns pada ANOVA Table:***
# 1. **sum_sq**: merepresentasikan jumlah kuadrat, ukuran variabilitas total dalam data. semakin tinggi nilai berarti semakin bervariable 
# 2. **df**: Degrees of freedom. ini mengindisikan jumlah independent values yang dapat bervariasi dalam analysis. total degrees of freedom adalah jumlah data point dikurang 1.
# 3. **F**: F-statistic mengukur ratio dari variance yang dijelaskan oleh model terhadap yang tidak dijelaskan oleh model. semakin tinggi semakin kuat relasi antara variable category dan variable continous.
# 4. **PR(>F)**:  p-value mengidikasikann probability bahwa hasil adalah kebetulan. p-values rendah (typically <0.05) menunjukkan bahwa efeknya significant secara statistic.
# 
# ***Interpretation of ANOVA Tables:***
# 1. **Make and Price**:
#    - **F**: 40.077596, **strong relationship** antara "Make" dan "Price".
#    - **PR(>F)**: 0.0, efek **signifikan** secara statistik.
# 
# 2. **Type and Price**:
#    - **F**: 9.895818, **significant relationship** antara "Type" dan "Price".
#    - **PR(>F)**: 0.0, efek **signifikan** secara statistik.
# 
# 3. **Origin and Price**:
#    - **F**: 56.316692, **strong relationship** antara "Origin" dan "Price".
#    - **PR(>F)**: 5.098152e-36, efek **signifikan** secara statistik.
# 
# 4. **Color and Price**:
#    - **F**: 9.170293, **significant relationship** antara "Color" dan "Price".
#    - **PR(>F)**: 2.151999e-20, efek **signifikan** secara statistik.
# 
# 5. **Options and Price**:
#    - **F**: 189.592482, **very strong relationship** antara "Options" dan "Price".
#    - **PR(>F)**: 3.511804e-81, efek **signifikan** secara statistik.
# 
# 6. **Fuel_Type and Price**:
#    - **F**: 0.812151, **weak relationship** antara "Fuel_Type" dan "Price".
#    - **PR(>F)**: 0.443939, efek **tidak signifikan** secara statistik.
# 
# 7. **Gear_Type and Price**:
#    - **F**: 161.578199, **very strong** relationship antara "Gear_Type" dan "Price".
#    - **PR(>F)**: 1.151752e-36, efek **signifikan** secara statistik.
# 
# 8. **Region and Price**:
#    - **F**: 14.239959, **strong relationship** antara "Region" dan "Price".
#    - **PR(>F)**: 5.466373e-61, efek **signifikan** secara statistik.
# 
# 9. **Negotiable and Price**:
#    - **F**: 2752.438765, **very strong relationshi**p antara "Negotiable" and "Price".
#    - **PR(>F)**: 0.0, efek **signifikan** secara statistik.
# 
# ***Summary***
# - **Significant Variables**: Variables like `Make`, `Type`, `Origin`, `Color`, `Options`, `Gear_Type`, `Region`, dan `Negotiable` memilike efek yang siginifikan secara statistic pada  `Price` (low p-values).
# - **Non-Significant Variables**: `Fuel_Type` tidak memiliki efek yang signifikan secara statistic pada `Price` (high p-value).
# 
# ini berarti Fuel Type dapat didrop pada saat data cleaning

# %% [markdown]
# ### Data Cardinalities

# # %%
# desc = df.describe().T
# desc['IQR'] = desc['75%'] - desc['25%']
# desc['median'] = df.median(numeric_only=True)
# desc['most_frequent'] = df.mode(numeric_only=True).iloc[0]
# desc

# # %%
# listTemp = df.select_dtypes(exclude=[float, int]).columns
# for i in listTemp:
#     print(f"Jumlah Unique Value dari {i}: {df[i].nunique()}")
#     print(df[i].unique(), "\n")

# # %%
# df.describe(exclude='number')

# %% [markdown]
# Make dan Type memiliki one to many relationship dimana satu brand pada Make memiliki berkaitan dengan beberapa model mobil pada Type.  
# ini berarti salah satu antara Make atau Type dapat didrop nantinya pada data cleaning.   
# 
# Negotiable dan Price berbanding terbalik dimana jika Price diatas 0 maka Negotiable pasti bernilai False, dan jika Negotiable bernilai True maka Price akan bernilai 0.  
# ini berarti Negotiable dapat didrop nantinya pada data cleaning. karena kita ingin memprediksi nilai dari Price. 
# 
# 

# %%
# df[(df['Price'] > 0) & (df['Negotiable'] == True)]

# # %% [markdown]
# # ### Identify missing value, outlier, anomaly, duplicates, etc

# # %% [markdown]
# # **Missing Value**

# # %%
# dataDesc = []
# for i in df.columns:
#     dataDesc.append([
#         i,
#         df[i].dtypes,
#         df[i].isna().sum(),
#         round((df[i].isna().sum() / len(df)) * 100, 2),
#         df[i].nunique(),
#         df[i].drop_duplicates().sample(2).values])
    
# pd.DataFrame(dataDesc, columns=[
#     "Data Features",
#     "Data Types",
#     "Null Val",
#     "Null Percentage",
#     "Unique",
#     "Unique Sample"])

# # %% [markdown]
# # dataset terlihat tidak memiliki missing value pada semua row dan columnya

# # %% [markdown]
# # **Outlier**

# # %%
# listTemp = df.select_dtypes(exclude=[float, int]).columns

# for i in listTemp:
#     frequency = df[i].value_counts()
#     outliers = frequency[frequency <= 1]
#     print(f"Outliers in {i}: {outliers.index.tolist()}")
#     print(f"Outliers number : {outliers.size}")
#     print(f"Outliers percentage :{outliers.sum() / df[i].shape[0]:.3f}%")
#     print()

# # %% [markdown]
# # pada column dengan value nominal/categorical outlier dapat dikatakan sebagai make atau brand yang rare i.e hanya ada 1 pada dataset

# # %%
# desc = df.describe().T
# desc['IQR'] = desc['75%'] - desc['25%']
# listTemp = desc.index.tolist()

# for i in listTemp:
#     outliers = df[(df[i] < (desc.loc[i]['25%'] - 1.5 * desc.loc[i]['IQR'])) | (df[i] > (desc.loc[i]['75%'] + 1.5 * desc.loc[i]['IQR']))]
#     print(f"outlier {i} berjumlah :{outliers.shape[0]}")
#     print(f"Outliers percentage :{outliers.shape[0] / df[i].shape[0]:.3f}%")
#     print(f"iqr \t: {desc.loc[i]['IQR']}")
#     print(f"lowerbound \t: {desc.loc[i]['25%'] - 1.5 * desc.loc[i]['IQR']:.1f}")
#     print(f"upperbound \t: {desc.loc[i]['75%'] + 1.5 * desc.loc[i]['IQR']:.1f}")
#     print()

# # %% [markdown]
# # column Year memiliki 357 outlier
# # dengan lowerbound pada Year 2023 dan upperbound pada 2027
# # 
# # column Engine_Size memiliki 37 outlier
# # dengan lowerbound pada Engine_Size -1.6~ dan upperbound pada 8.0
# # 
# # column Mileage memiliki 258 outlier
# # dengan lowerbound pada Mileage -200000.0 dan upperbound pada 432000.0
# # 
# # column Price memiliki 364 outlier
# # dengan lowerbound pada Price -110625.0 dan upperbound pada 184375.0
# # price harus dihitung ulang setelah data cleaning karena price bernilai 0 karena negotiable bernilai true masih pada data.

# # %%
# df.sort_values(by=['Mileage'], ascending=False)

# # %%
# df.sort_values(by=['Engine_Size'], ascending=False).head(37)

# # %% [markdown]
# # **Duplicate**

# # %%
# df.duplicated().sum()

# # %%
# df[df.duplicated(keep=False)]

# # %%
# df[df.duplicated(keep=False) & ~df.duplicated(keep='first')]

# # %% [markdown]
# # ditemukan 3 duplicate data yaitu pada,  
# # index 1354 dengan index 2027  
# # index 1387 dengan index 3201  
# # index 4272 dengan index 5070  
# # 
# # duplicate ini nantinya akan di drop pada data cleaning.

# # %% [markdown]
# # ## Preprocessing

# # %% [markdown]
# # ### Data Cleaning

# # %% [markdown]
# # - data tidak memiliki missing value
# # - Data Type pada data tidak perlu untuk diubah

# # %% [markdown]
# # - Removing Data Duplication

# # %%
# df.shape[0]

# # %%
# df[df.duplicated(keep=False) & ~df.duplicated(keep='first')]

# # %%
# df[~df.duplicated(keep='first')].shape[0]

# # %%
# dfRaw = df.copy(deep=True)
# df = df[~df.duplicated(keep='first')]
# df.shape

# # %% [markdown]
# # 3 data duplicate dihapus dari dataframe

# # %% [markdown]
# # - Inconsistent Variables / Outlier

# # %%
# desc = df.describe().T
# desc['IQR'] = desc['75%'] - desc['25%']
# listTemp = desc.index.tolist()

# for i in listTemp:
#     outliers = df[(df[i] < (desc.loc[i]['25%'] - 1.5 * desc.loc[i]['IQR'])) | (df[i] > (desc.loc[i]['75%'] + 1.5 * desc.loc[i]['IQR']))]
#     print(f"outlier {i} berjumlah :{outliers.shape[0]}")
#     print(f"Outliers percentage :{outliers.shape[0] / df[i].shape[0]:.3f}%")
#     print(f"iqr \t: {desc.loc[i]['IQR']}")
#     print(f"lowerbound \t: {desc.loc[i]['25%'] - 1.5 * desc.loc[i]['IQR']:.1f}")
#     print(f"upperbound \t: {desc.loc[i]['75%'] + 1.5 * desc.loc[i]['IQR']:.1f}")
#     print()

# # %%
# dfOutlierYear = df[(df['Year'] > 2027.0) | (df['Year'] < 2003.0)]
# df = df[~((df['Year'] > 2027.0) | (df['Year'] < 2003.0))]
# df.sort_values(by=['Year'], ascending=False)

# # %%
# df.shape

# # %%
# dfOutlierMileage = df[(df['Mileage'] > 432000.0) | (df['Mileage'] < -200000.0)]
# df = df[~((df['Mileage'] > 432000.0) | (df['Mileage'] < -200000.0))]
# df.sort_values(by=['Mileage'], ascending=False)

# # %%
# df.shape

# # %%
# dfOutlierEngine_Size = df[(df['Engine_Size'] > 8.0) | (df['Engine_Size'] < -1.6)]
# df = df[~((df['Engine_Size'] > 8.0) | (df['Engine_Size'] < -1.6))]
# df.sort_values(by=['Engine_Size'], ascending=False)

# # %%
# df.shape

# # %% [markdown]
# # **Outlier Year dapat didrop**, karena range lowerbound dan upperbound dari Year adalah 2003 sampai 2027, 24 tahun. walaupun maks tahun pada data adalah 2022, jadi real rangenya adalah 19 tahun. outlier Year juga hanya 0.044% dari keseluruhan data. Year aman untuk di drop karena kemungkinan untuk orang menjual mobil berusia 3-10 tahun sangat kecil[^1]
# # 
# # **Outlier Mileage juga dapat didrop**, karena range lowerbound dan uppperbound dari Mileage adalah -200000 sampai 432000, 632000 miles. walalaupun tidak memungkinkan Mileage dibawah 0, jadi rangenya adalah 432000. outlier Mileage juga hanya 0.032% dari keseluruhan data. Mileage aman untuk di drop karena kemungkinan untuk orang menjual mobil dengan mileage diatas 12000 kecil[^2]
# # 
# # **Outlier Engine_Size juga dapat di drop**, karena range lowerbound dan upperbound dari Engine_Size adalah -1.6 sampai 8.0, 9.6 liter. walaupun tidak memungkin Engine_Size dibawah 0, jadi rangenya adalah 8.0 liter. outlier Engine_Size juga hanya 0.005% dari keseluruhan data. Engine_Size aman untuk di drop karena kemungkinan adanya mobil diatas 8 liter sangat kecil dengan hanya ada 9 mobil consumer yang memiliki Engine_Size diatas 8.0 liter dari tahun 2003[^3]
# # 
# # selain itu pada pengecekan Engine_Size data dengan Engine_Size diatas 8.0, memiliki banyak data yang tidak akurat.   
# # beberapa contohnya adalah pada index :    
# # - 6835	Porsche	Cayenne Turbo GTS 2013 9.0(Engine_Size), yang mana seharusnya Engine_Size dari Porsche Cayenne Turbo GTS 2013 adalah 4.8
# # - 7504	Mercedes S 2006	9.0(Engine_Size), yang mana seharusnya Engine_Size dari Mercedes S 2006 adalah 3.0 - 6.0 (tergantung pada model)
# # - 3849	Toyota	Corolla	2001 8.1(Engine_Size), yang mana seharusnya Engine_Size dari Toyota	Corolla	2001 adalah 1.3 - 2.2 (Tergantung pada model)
# # 
# # [^1]: https://www.financialsamurai.com/the-ideal-length-of-time-to-own-a-car/
# # [^2]: https://motorway.co.uk/sell-my-car/guides/what-mileage-is-good-for-a-used-car
# # [^3]: https://www.autosnout.com/Cars-Engine-Size-List.php

# # %%
# df[(df['Price'] > 184375.0) | (df['Price'] < -110625.0)].shape[0]

# # %%
# df[(df['Price'] > 185000.0) | (df['Price'] < -55000.0)]['Year'].unique()

# # %%
# df[((df['Price'] > 185000.0) | (df['Price'] < -55000.0)) & (df['Year'] > 2018)].sort_values(by='Year', ascending=False).shape[0] / df[df['Year'] > 2018].shape[0]

# # %%
# df.sort_values(by=['Price'], ascending=False).head(30)

# # %%
# # dfOutlierEngine_Size = df[(df['Price'] > 450000) | (df['Price'] < 4000)]
# df = df[~(df['Price'] < 4000)]
# df.sort_values(by=['Price'], ascending=False)

# # %% [markdown]
# # **Outlier Price tidak Drop**, ini karena Outlier Price berisi mobil mewah dan mobil baru (kurang dari 5 tahun). dan outlier mobil baru memuat 13.5% dari keseluruhan mobil baru yang di jual, yang mana cukup significant.   
# # Pada column Price drop terjadi pada semua mobil bekas dengan harga dibawah 4000, ini dikarenakan setelah di research hampir tidak pada beberapa website jual mobil saudi, tidak ada mobil yang benar-benar dijual di bawah harga 4000. jika harga di dibawah 4000 biasanya itu bukan harga asli dari mobilnya, untuk mengetahui harga asli mobil-mobil ini diharuskan untuk melihat deskripsi atau mengcontact seller. contoh[^1][^2][^3]
# # 
# # [^1]:https://ksa.yallamotor.com/used-cars/hyundai/tucson/2014/used-hyundai-tucson-2014-dammam-1725493
# # [^2]:https://www.dubizzle.sa/en/ad/%D8%AA%D9%88%D9%8A%D9%88%D8%AA%D8%A7-%D9%81%D9%88%D8%B1%D8%AA%D8%B4%D9%86%D8%B1-2018-ID110391359.html
# # [^3]:https://www.dubizzle.sa/en/ad/porsche-911-2024-sports-ID110395136.html

# # %%
# df.shape

# # %%
# df = df.drop('Negotiable', axis=1)
# df.head(5)

# # %%
# df = df[~(df['Price'] == 0)]
# df.head()

# # %%
# df.shape

# # %% [markdown]
# # goalsnya adalah untuk memprediksi `Price` yang accurate, dan karena `Negotiable` berbanding terbalik dengan `Price`, dimana jika `Negotiable` memiliki nilai (>0 / True) maka `Price`nya merukapan 0. ini dikarenakan Negotiable digunakan untuk mobil yang harga pricenya perlu di negosiasikan langsung pada penjual. maka dari itu column `Negotiable` pada dataset perlu di drop.

# # %% [markdown]
# # ### Cleaned Data Check

# # %%
# df.head()

# # %%
# dataDesc = []
# for i in df.columns:
#     dataDesc.append([
#         i,
#         df[i].dtypes,
#         df[i].isna().sum(),
#         round((df[i].isna().sum() / len(df)) * 100, 2),
#         df[i].nunique(),
#         df[i].drop_duplicates().sample(2).values])
    
# pd.DataFrame(dataDesc, columns=[
#     "Data Features",
#     "Data Types",
#     "Null Val",
#     "Null Percentage",
#     "Unique",
#     "Unique Sample"])

# # %%
# desc = df.describe().T
# desc['IQR'] = desc['75%'] - desc['25%']
# desc['median'] = df.median(numeric_only=True)
# desc['most_frequent'] = df.mode(numeric_only=True).iloc[0]
# desc

# # %%
# df.describe(exclude='number')

# # %%
# df.to_csv("./Datasets/UsedCarsSA_Clean_EN_v2.csv", index=False)

# # %% [markdown]
# # #### Cleaned Data Distribution

# # %% [markdown]
# # **Price**

# # %%
# fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(12, 6))
# sns.boxplot(df['Price'], ax=ax_box)
# ax_box.set(title=f'{"Price"} box plot', xlabel='', ylabel='')    
# sns.histplot(df['Price'], ax=ax_hist, kde=True)
# ax_hist.set(title=f'{"Price"} hist plot', xlabel='', ylabel='')
# plt.tight_layout()
# plt.show()

# # %% [markdown]
# # **Numerical**

# # %%
# listTemp = ['Year','Engine_Size', 'Mileage']

# for i in listTemp:
#     fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(12, 6))
#     sns.boxplot(df[i], ax=ax_box)
#     ax_box.set(title=f'{i} (cleaned) box plot', xlabel='', ylabel='')    
#     sns.histplot(df[i], ax=ax_hist, kde=True)
#     ax_hist.set(title=f'{i} (cleaned) hist plot', xlabel='', ylabel='')
#     plt.tight_layout()
#     plt.show()

# # %% [markdown]
# # **Categorical**

# # %%
# listTemp = df.select_dtypes(exclude=[float, int]).columns
# for i in listTemp:
#     print(f"Jumlah Unique Value dari {i}: {df[i].nunique()}")
#     print(df[i].unique(), "\n")

# # %% [markdown]
# # ## Data Analysis

# # %% [markdown]
# # untuk membantu menaikkan revenue ada beberapa hal yang dapat dilakukan melalaui analysis. 
# # 1. Price Distribution
# # 2. Brand Popularity & price average by Region
# # 3. Gear Type & price average by Region
# # 4. Fuel Type & price average by Region
# # 5. Mileage and Age Relationship

# # %% [markdown]
# # ### Price Distribution

# # %%
# dfWOOutlierPrice = df[~((df['Price'] > 184375.0) | (df['Price'] < -110625.0))]
# fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(12, 6))
# sns.boxplot(dfWOOutlierPrice['Price'], ax=ax_box)
# ax_box.set(title=f'{"Price"} box plot', xlabel='', ylabel='')    
# sns.histplot(dfWOOutlierPrice['Price'], ax=ax_hist, kde=True)
# ax_hist.set(title=f'{"Price"} hist plot', xlabel='', ylabel='')
# plt.tight_layout()
# plt.show()

# print(f"Median :{dfWOOutlierPrice['Price'].median()}")
# print(f"Max : {dfWOOutlierPrice['Price'].max()}")
# print(f"Min : {dfWOOutlierPrice['Price'].min()}")

# # %% [markdown]
# # **Insight** : Tanpa Outlier *Price*  berada di range 500 - 183000, dengan Median pada 56254.  
# # **Recomendation** : mentargetkan segment *Price* yang berbeda dengan marketing strategies yang disesuaikan.   
# # **Action Item** : Mensegementasikan Price menjadi 3 yaitu, Budget atau dibawah 60000, Mid-Range atau diantara 60000 sampai dengan 185000 (batas outlier), dan High-end atau diatas 185000  
# #    
# # penerapan segementasi ini dapat dilakukan dengan rekomendasi mobil yang muncul pada home page tergantung pada history pencarian mobil customer pada Platform syarah.com.   
# # effectnya merupakan,    
# # - Meningkatkan Customer Satisfaction : Customer hanya diberikan rekomendasi pada sekitar price budgetnya. 
# # - Kemudahan Query Platform : Platform tidak perlu meload keseluruhan data melainkan hanya data yang ada pada segment yang telah ditentukan. 
# # - (Keuntungan revenue untuk perusahaan.)
# # 
# # keuntungan perusahaan : 
# # - menaikan revenue karena customer satisfaction yang tinggi, akan menambah/mempercepat transaksi pada platform

# # %% [markdown]
# # ### Brand Popularity and Region, rural, sub-urban, urban

# # %%
# df['Region'].unique()

# # %%
# data = {
#     'Riyadh' : 4137280, 
#     'Jeddah' : 2833169, 
#     'Dammam' : 5125254, 
#     'Al-Medina' : 2389452, 
#     'Qassim' : 1336179,
#     'Makkah': 2389452,
#     'Jazan': 1404997, 
#     'Aseer' : 2024285, 
#     'Al-Ahsa' : 908366, 
#     'Taef' : 885474, 
#     'Sabya' : 228375, 
#     'Al-Baha' : 339174, 
#     'Khobar' : 455541,
#     'Tabouk' : 886036, 
#     'Yanbu' : 250244, 
#     'Hail' : 746406, 
#     'Al-Namas' : 53908, 
#     'Jubail' : 224430, 
#     'Al-Jouf' : 595822, 
#     'Abha' : 1093705,
#     'Hafar Al-Batin' : 338636, 
#     'Najran' : 592300, 
#     'Arar' : 373577, 
#     'Besha' : 202096, 
#     'Qurayyat' : 125090, 
#     'Sakaka' : 595822,
#     'Wadi Dawasir' : 92631,
# }

# df_region_population = pd.DataFrame(list(data.items()), columns=['Region', 'Population'])
# df_region_population

# # %% [markdown]
# # Population Data gathered from,    
# # https://en.wikipedia.org/wiki/Provinces_of_Saudi_Arabia    
# # https://en.wikipedia.org/wiki/Subdivisions_of_Saudi_Arabia   

# # %% [markdown]
# # Dataset tidak memiliki jumlah Populasi atau categorisasi yang berkaitan dengan Regionnya. Region pada dataset juga tidak sama dengan region yang tertulis pada database official region dari saudi arabia, dimana pada officialnya hanya terdapat 13 region (provinces of sadu arabia) sementara terdapat 27 unique region pada dataset. data populasi diambil dari beberapa sumber yang berbeda, karena tidak ada satu sumber yang memiliki keseluruhan data populasi, mengingat sebagian adalah provinsi dan sebagian adalah city/kota. 

# # %%
# kmeans = KMeans(n_clusters=3, random_state=0).fit(df_region_population[['Population']])
# df_region_population['Cluster'] = kmeans.labels_
# category_map = {0: 'rural', 1: 'urban', 2: 'suburban'}
# df_region_population['Category'] = df_region_population['Cluster'].map(category_map)

# # %%
# df_region_population.sort_values(by='Population', ascending=False)

# # %%
# plt.scatter(df_region_population['Region'], df_region_population['Population'], c=df_region_population['Cluster'])
# plt.xlabel('Region')
# plt.ylabel('Population')
# plt.title('Region Clustering')
# plt.xticks(rotation=75)
# plt.show()

# # %% [markdown]
# # KMeans Clustering digunakan untuk men-categorisasikan region dengan nilai populasinya, 3 category digunakan untuk merepresentasikan Urban, Sub-Urban, dan Rural. dimana dari hasil clustering, Urban memiliki populasi diatas, +-400.000, Sub-Urban berada pada +- 120.000 sampai dengan 300.000, dan Rural berada pada populasi dibawah +- 120.000.

# # %%
# merged_df = pd.merge(df, df_region_population[['Region', 'Category']], on='Region', how='left')
# merged_df

# # %%
# grouped_df = merged_df.groupby(by=['Category', 'Make'])[['Price']].count().reset_index()
# grouped_df.rename(columns={'Price': 'Count'}, inplace=True)
# top_5_df = grouped_df.groupby('Category')[['Category', 'Make', 'Count']].apply(lambda x: x.nlargest(5, 'Count')).reset_index(drop=True)
# top_5_df

# # %%
# print(f"jumlah top 5 : {top_5_df['Count'].sum()}")
# print(f"Jumlah top 5 percentage dari keseluruhan {(top_5_df['Count'].sum()/df.shape[0]) * 100:.2f}%")

# # %% [markdown]
# # **Insight** : 5 Terbesar (Toyota, Hyundai, Ford, Chevrolet, Nissan)berjumlah 3107, yaitu 59.41% dari keseluran total.  
# # **Recomendation** : Fokus pada Brand ini untuk promosi dan penampilan pada homepage.   
# # **Action Item** : Buat bagian khusus dan penawaran promosi untuk Brand populer pada Platform tergantung pada region mana platform di access. 
# # 
# # Pemasaran pada Brand populer menjangkau lebih banyak potential customer, dibandingkan dengan mempromosikan Brand yang tidak banyak dikenal oleh orang-orang.  
# # 
# # keuntungan perusahaan :   
# # customer marketing strategies dapat menaikkan platform brand dan customer loyalty dan membantu customer retantion dan juga mempercepat penjualan. 5% increase pada customer retention dapat menjadi 25% 95% increase pada profit, existing buyer mengeluarkan upto 300% dari yang baru, memakan 5x lipat lebih banyak uang untuk mendapatkan customer baru daripada me-retain yang sudah ada.[^1]
# # 
# # [^1]: https://xgrowth.com.au/blogs/what-is-customer-marketing

# # %% [markdown]
# # ### Type Popularity and Region, rural, sub-urban, urban with price average

# # %%
# grouped_df = merged_df.groupby(by=['Category', 'Type'])[['Price']].count().reset_index()
# grouped_df.rename(columns={'Price': 'Count'}, inplace=True)
# top_3_df = grouped_df.groupby('Category')[['Category', 'Type', 'Count']].apply(lambda x: x.nlargest(3, 'Count')).reset_index(drop=True)
# top_3_df

# # %% [markdown]
# # **Insight** : 7 Terbesar (Accent, Land Cruiser, Taurus, Camry, Sonata, Elantra, Hilux)    
# # **Recomendation** : Fokus pada Make ini untuk promosi dan penampilan pada homepage pada categorynya masing-masing.      
# # **Action Item** : Buat bagian khusus dan penawaran promosi untuk Type populer pada Platform tergantung pada region mana platform di access. 
# # 
# # Pemasaran pada Type populer menjangkau lebih banyak potential customer dari region-region tersebut, dibandingkan dengan mempromosikan Brand yang tidak popular oleh pada region tersebut.  
# # 
# # keuntungan perusahaan :   
# # perusahaan juga dapat menjuaal feature specified region advertisement, untuk seller yang mau memasang ads/promoted untuk mobilnya pada suatu region tersendiri
# # 
# # customer marketing strategies dapat menaikkan platform brand dan customer loyalty dan membantu customer retantion dan juga mempercepat penjualan. 5% increase pada customer retention dapat menjadi 25% 95% increase pada profit, existing buyer mengeluarkan upto 300% dari yang baru, memakan 5x lipat lebih banyak uang untuk mendapatkan customer baru daripada me-retain yang sudah ada.[^1]
# # 
# # [^1]: https://xgrowth.com.au/blogs/what-is-customer-marketing

# # %% [markdown]
# # ### Gear Type and Region, rural urban with price average

# # %%
# grouped_df = merged_df.groupby(by=['Category', 'Gear_Type']).agg(
#     Price_Average=('Price', 'mean'),
#     Count=('Price', 'count')
# ).reset_index()
# total_counts = grouped_df.groupby('Category')['Count'].sum().reset_index(name='Total_Count')
# grouped_df = pd.merge(grouped_df, total_counts, on='Category')
# grouped_df['Percentage'] = (grouped_df['Count'] / grouped_df['Total_Count']) * 100
# grouped_df.drop(columns=['Total_Count'])

# # %%
# print(f"rata-rata perbedaan harga automatic dan manual : {(grouped_df[grouped_df['Gear_Type'] == 'Automatic']['Price_Average'].sum()-grouped_df[grouped_df['Gear_Type'] == 'Manual']['Price_Average'].sum())/3:.2f}")

# # %% [markdown]
# # **Insight** : Automatic memiliki Percentase yang lebih tinggi dari manual pada setiap category, yaitu rural 87.3%, suburban 90.4%, dan urban 91.9%. rata-rata harga automatic juga leibh mahal 23.186,87 dari harga manual. (mengutamakan kenyamanan meskipun secara price lebih mahal)  
# # **Recomendation** : marketing gear_type tertentu tergantung pada harga yang dicari customer.    
# # **Action Item** : menampilkan lebih banyak mobil bekas dengan automatic Gear_Type pada customer yang mencari dengan budget diatas price_average mobil manual pada setiap categorynya. dan menawarkan lebih banyak mobil bekas dengan manual Gear_Type pada customer yang mencari dengan budget dibawah price_average mobil mnual pada setiap categorynya.      
# #     
# # Data akan lebih berguna jika ditambahkan dengan dataset populasi per region dan dataset history pencarian/query pada platform. dengan penambahan data ini, persebaran data mobil bekas setiap region category-nya akan lebih akurat, dan dataset history pencarian/query akan membantu mengetahui apa yang sebenarnya dicari oleh customer pada region-region tersebut, dan kita bisa menyesuaikan mobil yang di-highlight pada platform sesuai dengan regionnya
# # 
# # keuntungan perusahaan :  
# # customer marketing strategies dapat menaikkan platform brand dan customer loyalty dan membantu customer retantion dan juga mempercepat penjualan. 5% increase pada customer retention dapat menjadi 25% 95% increase pada profit, existing buyer mengeluarkan upto 300% dari yang baru, memakan 5x lipat lebih banyak uang untuk mendapatkan customer baru daripada me-retain yang sudah ada.[^1]
# # 
# # [^1]: https://xgrowth.com.au/blogs/what-is-customer-marketing

# # %% [markdown]
# # ### Fuel Type and Region, rural urban with price average

# # %%
# grouped_df = merged_df.groupby(by=['Fuel_Type', 'Category']).agg(
#     Price_Average=('Price', 'mean'),
#     Count=('Price', 'count')
# ).reset_index()
# total_counts = grouped_df.groupby('Fuel_Type')['Count'].sum().reset_index(name='Total_Count')
# grouped_df = pd.merge(grouped_df, total_counts, on='Fuel_Type')
# grouped_df['Percentage'] = (grouped_df['Count'] / grouped_df['Total_Count']) * 100
# grouped_df.drop(columns=['Total_Count']).sort_values(by=['Fuel_Type', 'Category', 'Count'], ascending=[True, False, False])

# # %% [markdown]
# # **Insight** : mobil bekas dengan Fuel_Type gas jauh lebih populer pada area urban, Fuel_Type Diesel populer pada Urban dan Sub-Urban, dan Fuel_Type Hybrid pada area urban.     
# # **Recomendation** : men-update tampilan listing mobil bekas pada region populernya.    
# # **Action Item** : dengan memberikan badge, flaire, atau label pada mobil dengan Fuel_Type selain gas.      
# #    
# # customer yang mencari mobil dengan type tersebut akan lebih mudah menemukan mobil yang diinginkan, mobil yang lebih populer di daerahnya akan lebih cepat laku. yang dapat meningkatkan customer satisfaction, dan conversion rate.
# # 
# # keuntungan perusahaan :  
# # customer marketing strategies dapat menaikkan platform brand dan customer loyalty dan membantu customer retantion dan juga mempercepat penjualan. 5% increase pada customer retention dapat menjadi 25% 95% increase pada profit, existing buyer mengeluarkan upto 300% dari yang baru, memakan 5x lipat lebih banyak uang untuk mendapatkan customer baru daripada me-retain yang sudah ada.[^1]
# # 
# # [^1]: https://xgrowth.com.au/blogs/what-is-customer-marketing

# # %% [markdown]
# # ### Mileage and Age Relationship

# # %%
# plt.figure(figsize=(10,8))
# sns.heatmap(df.corr(method='spearman', numeric_only=True), annot=True)
# plt.title('Numerical Correlation')
# plt.show()

# # %% [markdown]
# # **Insight** : ada Strong Negative Correlatioin antara Umur Mobil dan Mileagenya, mobil yang lebih tua cenderung memiliki milage yang lebih besar)    
# # **Recommendation** : Highlight mobil dengan jarak tempuh rendah sebagai nilai jual untuk model lama.     
# # **Action Item** : mengimplementasikan filter dan label mobil "Mileage Rendah" dan "Produksi Baru".    
# #    
# # mobil yang lebih tua dengan mileage yang lebih rendah berarti mobil tersebut adalah mobil secondary / jarang digunakan yang biasanya lebih banyak dicari daripada mobil dengan mileage yang tinggi. 
# # 
# # keuntungan perusahaan :     
# # customer yang mencari mobil pada tahun tertentu tidak perlu mencari dengan susah dari listing untuk mencari mobil yang memiliki mileage yang rendah. menaikkan/mempercepat transaksi pada platform
# # 
# # customer marketing strategies dapat menaikkan platform brand dan customer loyalty dan membantu customer retantion dan juga mempercepat penjualan. 5% increase pada customer retention dapat menjadi 25% 95% increase pada profit, existing buyer mengeluarkan upto 300% dari yang baru, memakan 5x lipat lebih banyak uang untuk mendapatkan customer baru daripada me-retain yang sudah ada.[^1]
# # 
# # [^1]: https://xgrowth.com.au/blogs/what-is-customer-marketing
# # 

# # %% [markdown]
# # ### Mileage and Age Relation dengan Popular Type

# # %%
# median_mileage = merged_df['Mileage'].median()
# merged_df['Mileage_Category'] = merged_df['Mileage'].apply(lambda x: 'high' if x > median_mileage else 'low')
# merged_df

# # %%
# grouped_df = merged_df.groupby(by=['Mileage_Category', 'Category']).agg(
#     Price_Average=('Price', 'mean'),
#     Count=('Price', 'count')
# ).reset_index()
# total_counts = grouped_df.groupby('Category')['Count'].sum().reset_index(name='Total_Count')
# grouped_df = pd.merge(grouped_df, total_counts, on='Category')
# grouped_df['Percentage'] = (grouped_df['Count'] / grouped_df['Total_Count']) * 100
# grouped_df.drop(columns=['Total_Count']).sort_values(by=['Category', 'Mileage_Category','Count'], ascending=[False, False, False])

# # %%
# median_mileage = merged_df['Year'].median()
# merged_df['Age_Category'] = merged_df['Year'].apply(lambda x: 'newer' if x > median_mileage else 'older')
# grouped_df = merged_df.groupby(by=['Age_Category', 'Category']).agg(
#     Price_Average=('Price', 'mean'),
#     Count=('Price', 'count')
# ).reset_index()
# total_counts = grouped_df.groupby('Category')['Count'].sum().reset_index(name='Total_Count')
# grouped_df = pd.merge(grouped_df, total_counts, on='Category')
# grouped_df['Percentage'] = (grouped_df['Count'] / grouped_df['Total_Count']) * 100
# grouped_df.drop(columns=['Total_Count']).sort_values(by=['Category', 'Age_Category', 'Count'], ascending=[False, False, False])

# %% [markdown]
# **Insight** : percentage jumlah mileage tinggi dan rendah tidak jauh berbeda pada setiap categorynya. tetapi harga averagenya miliki perbedaan sekitar 40%. percentage age baru dan lama tidak memiliki perbedaan yang significan pada urban, memiliki perbedaan yang significan pada suburban, dan sangat significant pada  rural, dengan perbedaan harga average yang sama dengan mileage     
# **recomendation** : men-highlight mobil-mobil tertentu pada platform dengan mileage dan age terhadap price-average dan popularitynya sesuai dengan criteria categorynya     
# **action** : Highlight mobil dengan jarak tempuh rendah sebagai nilai jual untuk model lama, terutama dengan harga yang cenderung murah (dibawah price_average-nya). highlight mobil dengan age tua dengan mileage pada suburban dan rural untuk customer yang mencari dibawah price average cateogorynya, dan highlight mobil dengan age yang rendah jika customer mencari mobil bekas dengan harga yang lebih tinggi dari price_averagenya. 
# 
# keuntungan perusahaan :     
# used cars suggestion yang lebih disesuaikan pada type customer tertentu akan dapat menaikkan/mempercepat transaksi pada platform
# 
# customer marketing strategies dapat menaikkan platform brand dan customer loyalty dan membantu customer retantion dan juga mempercepat penjualan. 5% increase pada customer retention dapat menjadi 25% 95% increase pada profit, existing buyer mengeluarkan upto 300% dari yang baru, memakan 5x lipat lebih banyak uang untuk mendapatkan customer baru daripada me-retain yang sudah ada.[^1]
# 
# [^1]: https://xgrowth.com.au/blogs/what-is-customer-marketing

# %% [markdown]
# ## Machine Learning

# %% [markdown]
# ### Feature Engineering

# %%
df = pd.read_csv("./Datasets/UsedCarsSA_Clean_EN_v2.csv")
df

# %% [markdown]
# Make dan Type, memiliki hubungan one to many, setiap satu Make memiliki satu atau lebih Type, dan setiap Type memiliki satu Make (2 atau lebih Type yang berbeda bisa memiliki Make yang sama), jadi tidak dibutuhkan dua2nya pada saat training model.    
# 
# maka training akan dipisah menjadi dua yaitu dengan Make dan dengan Type. 
# 

# %% [markdown]
# **Table**

# %%
columns = ['model_name', 'MedAE_train', 'MedAE_Test', 'MedAE_diff', 'MAE_train', 'MAE_Test', 'MAE_diff', "MAPE_train", "MAPE_Test"]
scores = pd.DataFrame(columns=columns)
scores

# %% [markdown]
# pembuatan table untuk menyimpan score dari hasil machine learning

# %% [markdown]
# #### Feature and Target

# %%
X = df.drop(columns=['Type','Price'])
X_Type = df.drop(columns=['Make','Price'])
y = df['Price']

# %% [markdown]
# feature yang digunakan sebagai base (X) adalah 'Make', 'Year', 'Origin', 'Color', 'Options', 'Engine_Size', 'Fuel_Type', 'Gear_Type', 'Mileage', 'Region'.     
# feature yang digunakan sebagai secondary (X_Type) adalah 'Type', 'Year', 'Origin', 'Color', 'Options', 'Engine_Size', 'Fuel_Type', 'Gear_Type', 'Mileage', 'Region'.     
# dan Targetnya adalah Price  

# %% [markdown]
# #### Data Splitting

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=42)

# %% [markdown]
# data di split menjadi train dan test dengan ration 80/20

# %% [markdown]
# ### Preprocessor (Encoding and Scalling)

# %% [markdown]
# Ordinal Encoder : 'Options' (memiliki tingkatan)  
# One Hot Encoder : 'Make', 'Origin', 'Color', 'Fuel_Type', 'Gear_Type', 'Region'   
# MinMax Scaller : 'Year', 'Engine_Size' (outlier lebih kecil)    
# Robust Scaler : 'Mileage' (Kemungkinan outlier lebih besar)   

# %% [markdown]
# 'option' menggunakan ordinal encoder karena memiliki tingkatan     
# 'Make', 'Origin', 'Color', 'Fuel_Type', 'Gear_Type', 'Region' menggunakan one hot encoder karena merupakan nominal column tanpa tingkatan khusus     
# 'Year', 'Engine_Size' menggunakan MinMax Scaller karena memiliki outlier yang tidak terlalu significan    
# 'Mileage' Menggunakan Robust Scaller karena memilik outlier yang lebih significant
# 

# %%
df.columns

# %%
df['Options'].unique()
categories = [['Full', 'Semi Full', 'Standard']]
categories

# %%
preprocessor = ColumnTransformer([
    ('OE', OrdinalEncoder(categories=categories), ['Options']),
    ('OH', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), ['Make', 'Origin', 'Color', 'Fuel_Type', 'Gear_Type', 'Region']),
    ('MMS', MinMaxScaler(), ['Year', 'Engine_Size']),
    ('RS', RobustScaler(), ['Mileage']),
])

# %%
preprocessornd = ColumnTransformer([
    ('OE', OrdinalEncoder(categories=categories), ['Options']),
    ('OH', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), ['Make', 'Origin', 'Color', 'Fuel_Type', 'Gear_Type', 'Region']),
    ('MMS', MinMaxScaler(), ['Year', 'Engine_Size']),
    ('RS', RobustScaler(), ['Mileage']),
])

# %% [markdown]
# dibuat 2 type preprocessor, yaitu:
# - 'preprocessor' : menggunakan drop first pada onehot encodernya untuk linear type regression.
# - 'preprocessornd' : tidak melakukan drop first pada onehot encodernya untuk tree dan knn type regression.

# %% [markdown]
# ### benchmarking K-Fold untuk data train

# %%
# models = [LinearRegression(),Lasso(max_iter=10000),Ridge(),KNeighborsRegressor(),DecisionTreeRegressor(),RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor(), LGBMRegressor(force_col_wise=True)]
# score=[]
# rata=[]
# std=[]

# for i in models:
#     skfold=KFold(n_splits=5)
#     estimator=Pipeline([
#         ('preprocess',preprocessornd),
#         ('model',i)])
#     model_cv=cross_val_score(estimator,X_train,y_train,cv=skfold,scoring='r2', error_score='raise')
#     score.append(model_cv)
#     rata.append(model_cv.mean())
#     std.append(model_cv.std())
    
# pd.DataFrame({'model':['Linear Regression', 'Lasso', 'Ridge', 'KNeighbors', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM'],'mean r2_score':rata,'sdev':std}).set_index('model').sort_values(by='mean r2_score',ascending=False)

# %% [markdown]
# ini berarti,    
# XGBoost memiliki performance terbaik dengan score r2 8.029199e-01 dengan sdev 5.301794e-02  
# tetapi, Gradient Boosting lebih stabil dengan score r2 7.836736e-01	sdev 2.670416e-02   
# untuk mengecheck ulang maka kita akan mencoba semua model dan beserta tuningnya   

# %% [markdown]
# ### Linear Regression

# %%
# LinRegModel = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regressor', LinearRegression())
# ])

# # Fit the pipeline on the training data
# LinRegModel.fit(X_train, y_train)

# # %%
# y_pred_tr = LinRegModel.predict(X_train)
# y_pred_ts = LinRegModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr)
# medae_test = median_absolute_error(y_test, y_pred_ts)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr)
# mae_test = mean_absolute_error(y_test, y_pred_ts)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr) 
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts)

# row = {
#     'model_name': 'Linear Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)
               
# print(f"""
#     Linear Regression : 
#     train MedAE score : {median_absolute_error(y_train, y_pred_tr)}
#     test MedAE score : {median_absolute_error(y_test, y_pred_ts)}
#     train MAE Score : {mean_absolute_error(y_train, y_pred_tr)}]
#     test MAE Score : {mean_absolute_error(y_test, y_pred_ts)}

# """)

# # %% [markdown]
# # Linear regression dilakukan untuk mendapatkan base model.    
# # Linear Regression adalah technique machine learning datsar yang memodelkan relationship sebuah dependent variable kepada satu atau lebih independent variables menggunakan sebuah linear equation (𝑦 = 𝑚𝑥 + 𝑐)   
# # y : garis titik yang dihasilkan (dependant variable)
# # m : slope dari garis (coefficient)
# # x : independent variable
# # c : intercept (nilai y jika x adalah 0)

# # %% [markdown]
# # ### Lasso

# # %%
# LassoModel = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regressor', Lasso())
# ])

# # Fit the pipeline on the training data
# LassoModel.fit(X_train, y_train)

# # %%
# y_pred_tr_lasso = LassoModel.predict(X_train)
# y_pred_ts_lasso = LassoModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_lasso)
# medae_test = median_absolute_error(y_test, y_pred_ts_lasso)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_lasso)
# mae_test = mean_absolute_error(y_test, y_pred_ts_lasso)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_lasso)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_lasso)

# row = {
#     'model_name': 'Lasso Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     Lasso Regression : 
#     train MedAE score : {median_absolute_error(y_train, y_pred_tr_lasso)}
#     test MedAE score : {median_absolute_error(y_test, y_pred_ts_lasso)}
#     train MAE Score : {mean_absolute_error(y_train, y_pred_tr_lasso)}]
#     test MAE Score : {mean_absolute_error(y_test, y_pred_ts_lasso)}

# """)

# # %% [markdown]
# # Lasso model adalah linear regression method dengan regularisasi L1, yang menambahkan penalty yang setara dengan nilai absulot dari kebesaran keofisiennya. hal ini membuat model yang lebih sederhana dengan lebih sedikit parameter, yang secara efektif menjalankan pemilihan feature dengan mengecilkan beberapa koefisiesn meanjadi nol. model lasso berguna terutama untuk data berdimensi tinggi. 

# # %% [markdown]
# # ### Ridge

# # %%
# RidgeModel = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regressor', Ridge())
# ])

# # Fit the pipeline on the training data
# RidgeModel.fit(X_train, y_train)

# # %%
# y_pred_tr_ridge = RidgeModel.predict(X_train)
# y_pred_ts_ridge = RidgeModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_ridge)
# medae_test = median_absolute_error(y_test, y_pred_ts_ridge)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_ridge)
# mae_test = mean_absolute_error(y_test, y_pred_ts_ridge)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_ridge)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_ridge)

# row = {
#     'model_name': 'Ridge Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     Ridge Regression : 
#     train MedAE score : {median_absolute_error(y_train, y_pred_tr_ridge)}
#     test MedAE score : {median_absolute_error(y_test, y_pred_ts_ridge)}
#     train MAE Score : {mean_absolute_error(y_train, y_pred_tr_ridge)}]
#     test MAE Score : {mean_absolute_error(y_test, y_pred_ts_ridge)}

# """)

# # %% [markdown]
# # ridge model adalah metode regresi linear dengan regularisasi L2, yang menambahkan penalti dengan kuadrat besarnya koefisien. in imembantu mencegah overfitting dengan mengecilkan koefisien, tidak seperti lasso, model ini tidak menetapkan koefisien apapun secara tepat ke nol, sehingga cocok untuk situasi dimana berpotensi penting. sama seperti lasso, ridge juga digunakan untuk meningkatkan kinerja dan interpretablitas model, terutama dalam data berdimensi tinggi. 

# # %% [markdown]
# # 
# # ### Tuned Lasso

# # %%
# TunedLassoModel = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regressor', Lasso())
# ])

# skf = KFold(n_splits=5, random_state=42, shuffle=True)
# param_Lasso = { 
#     'regressor__alpha': [0.001, 0.01, 1, 10, 20, 30, 40, 50, 100], 
#     'regressor__max_iter': [1000, 2000, 3000],
# }

# scoring = { 'Median_Absolute_Error': 'neg_median_absolute_error', 'Mean_Absolute_Error': 'neg_mean_absolute_error' }

# GSCV_Lasso = GridSearchCV(TunedLassoModel, param_Lasso, cv=skf, scoring=scoring, refit='Median_Absolute_Error', n_jobs=12)
# GSCV_Lasso.fit(X_train, y_train)

# # %% [markdown]
# # tuning parameter pada lasso,
# # - alpha : constant yang mengalikan L1, mengendalikan kekuatan regularisasi, harus berupa float non-negatif
# # - max_iter : maximum number of iteration

# # %%
# GSCV_Lasso.best_score_

# # %%
# pd.DataFrame(GSCV_Lasso.cv_results_)[pd.DataFrame(GSCV_Lasso.cv_results_)['rank_test_Median_Absolute_Error'] == 1][['params', 'mean_test_Median_Absolute_Error']]

# # %%
# best_params = {k.split('__')[1]: v for k, v in GSCV_Lasso.best_params_.items() if k.split('__')[1] in ['alpha', 'max_iter']}

# BestLassoModel = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regressor', Lasso(**best_params))
# ])

# # Fit the pipeline on the training data
# BestLassoModel.fit(X_train, y_train)

# # %%
# y_pred_tr_lasso_b = BestLassoModel.predict(X_train)
# y_pred_ts_lasso_b = BestLassoModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_lasso_b)
# medae_test = median_absolute_error(y_test, y_pred_ts_lasso_b)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_lasso_b)
# mae_test = mean_absolute_error(y_test, y_pred_ts_lasso_b)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_lasso_b)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_lasso_b)

# row = {
#     'model_name': 'Tuned Lasso Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     Tuned Lasso Regression : 
#     train MedAE score : {median_absolute_error(y_train, y_pred_tr_lasso_b)}
#     test MedAE score : {median_absolute_error(y_test, y_pred_ts_lasso_b)}
#     train MAE Score : {mean_absolute_error(y_train, y_pred_tr_lasso_b)}
#     test MAE Score : {mean_absolute_error(y_test, y_pred_ts_lasso_b)}
# """)

# # %% [markdown]
# # ### Tuned Ridge

# # %%
# TunedRidgeModel = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regressor', Ridge())
# ])

# skf = KFold(n_splits=5, random_state=42, shuffle=True)
# param_Ridge = { 
#     'regressor__alpha': [0.001, 0.01, 1, 10, 20, 30, 40, 50, 100], 
#     'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'], 
#     'regressor__random_state' : [42],
# }

# scoring = { 'Median_Absolute_Error': 'neg_median_absolute_error', 'Mean_Absolute_Error': 'neg_mean_absolute_error' }

# GSCV_Ridge = GridSearchCV(TunedRidgeModel, param_Ridge, cv=skf, scoring=scoring, refit='Median_Absolute_Error', n_jobs=12)
# GSCV_Ridge.fit(X_train, y_train)

# # %% [markdown]
# # tuning parameter pada ridge,
# # - alpha : constant yang mengalikan L2, mengendalikan kekuatan regularisasi, harus berupa float non-negatif
# # - max_solver : solver yang digunakan untuk computasi,
# #     - auto : memilih secara otomatis
# #     - svd : sigular value decomposition dari X untuk menghitung coefficient ridge. solver palinng stabil, tetapi rada lambat. 
# #     - cholesky : menggunakan scipy.linalg.solve function untuk mendapatkan solusi closed-form
# #     - lsqr : menggunakan scipy.sparse.linalg.lsqr. sover paling cepat dan menggunakan iterative procedure. 
# #     - sparse_cg : menggunakan scipy.sparse.linalg.cd. sebagai algorithm iterative, solver yang lebih appropriate untuk large-scale data
# #     - sag : iterative procedure yang biasanya lebih cepat dari sover lainnya ketika sample dan featurenya besar. 
# #     - saga : improved version of sag
# # - random_state : digunakan oleh sover tertentu untuk men-shuffle data

# # %%
# GSCV_Ridge.best_score_

# # %%
# pd.DataFrame(GSCV_Ridge.cv_results_)[pd.DataFrame(GSCV_Ridge.cv_results_)['rank_test_Median_Absolute_Error'] == 1][['params', 'mean_test_Median_Absolute_Error']]

# # %%
# best_params = {k.split('__')[1]: v for k, v in GSCV_Ridge.best_params_.items() if k.split('__')[1] in ['alpha', 'solver', 'random_state']}

# BestRidgeModel = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regressor', Ridge(**best_params))
# ])

# # Fit the pipeline on the training data
# BestRidgeModel.fit(X_train, y_train)

# # %%
# y_pred_tr_ridge_b = BestRidgeModel.predict(X_train)
# y_pred_ts_ridge_b = BestRidgeModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_ridge_b)
# medae_test = median_absolute_error(y_test, y_pred_ts_ridge_b)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_ridge_b)
# mae_test = mean_absolute_error(y_test, y_pred_ts_ridge_b)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_ridge_b)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_ridge_b)

# row = {
#     'model_name': 'Tuned Ridge Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     Tuned Ridge Regression : 
#     train MedAE score : {median_absolute_error(y_train, y_pred_tr_ridge_b)}
#     test MedAE score : {median_absolute_error(y_test, y_pred_ts_ridge_b)}
#     train MAE Score : {mean_absolute_error(y_train, y_pred_tr_ridge_b)}
#     test MAE Score : {mean_absolute_error(y_test, y_pred_ts_ridge_b)}

# """)

# # %% [markdown]
# # ### Decision Tree Regressor

# # %%
# DecisionTreeModel = Pipeline([
#     ('preprocessor', preprocessornd),
#     ('regressor', DecisionTreeRegressor())
# ])

# # Fit the pipeline on the training data
# DecisionTreeModel.fit(X_train, y_train)

# # %% [markdown]
# # decision tree regressor adalah non parametric supervised learning algorithm untuk regresi. bekerja dengna membagi data menjadi subset-subset sesuai dengan feature value, membentuk tree like model keputusan. setiap internal node merepresentasikan sebuah keputusan berdasarkan feature, dan setiap leaf node merepresentasikan, output yang di produksi.goalnya adalah untuk meminimalisir mean squared error untuk mendapatkan best fit. 

# # %%
# y_pred_tr_dt = DecisionTreeModel.predict(X_train)
# y_pred_ts_dt = DecisionTreeModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_dt)
# medae_test = median_absolute_error(y_test, y_pred_ts_dt)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_dt)
# mae_test = mean_absolute_error(y_test, y_pred_ts_dt)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_dt)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_dt)

# row = {
#     'model_name': 'Decisision Tree Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     Decisision Tree Regression : 
#     train MedAE score : {medae_train}
#     test MedAE score : {medae_test}
#     train MAE Score : {mae_train}
#     test MAE Score : {mae_test}
# """)

# # %%
# scores

# # %% [markdown]
# # ### Tuned Decision Tree Regressor

# # %%
# TunedDecisionTreeModel = Pipeline([
#     ('preprocessor', preprocessornd),
#     ('regressor', DecisionTreeRegressor())
# ])

# param_DTR = {
#     'regressor__max_depth': list(range(2, 150)) + [None],
#     'regressor__min_samples_split': range(2, 21),
#     'regressor__min_samples_leaf': range(1,21),
#     'regressor__random_state' : [42],
# }

# scoring = { 'Median_Absolute_Error': 'neg_median_absolute_error', 'Mean_Absolute_Error': 'neg_mean_absolute_error' }

# GSCV_DT = GridSearchCV(TunedDecisionTreeModel, param_DTR, cv=skf, scoring=scoring, refit='Median_Absolute_Error', n_jobs=12)
# GSCV_DT.fit(X_train, y_train)

# # %% [markdown]
# # tuning parameter pada decision tree,
# # - max_depth : kedalaman(depth) maximum dari tree. 
# # - min_samples_split : jumlah minimum sample yang diperlukan untuk memisahkan sebuah internal node
# # - min_samples_leaf : jumlah minimum sample yang diperlukan untuk membuat sebuah leaf node
# # - random_state : mengontrol ke-random-an dari estimator. 

# # %%
# GSCV_DT.best_score_

# # %%
# pd.DataFrame(GSCV_DT.cv_results_)[pd.DataFrame(GSCV_DT.cv_results_)['rank_test_Median_Absolute_Error'] == 1][['params', 'mean_test_Median_Absolute_Error']]

# # %%
# best_params = {k.split('__')[1]: v for k, v in GSCV_DT.best_params_.items() if k.split('__')[1] in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state']}

# BestDecisionTreeModel = Pipeline([
#     ('preprocessor', preprocessornd),
#     ('regressor', DecisionTreeRegressor(**best_params))
# ])

# # Fit the pipeline on the training data
# BestDecisionTreeModel.fit(X_train, y_train)

# # %%
# y_pred_tr_dt_b = BestDecisionTreeModel.predict(X_train)
# y_pred_ts_dt_b = BestDecisionTreeModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_dt_b)
# medae_test = median_absolute_error(y_test, y_pred_ts_dt_b)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_dt_b)
# mae_test = mean_absolute_error(y_test, y_pred_ts_dt_b)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_dt_b)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_dt_b)

# row = {
#     'model_name': 'Tuned Decision Tree Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     Tuned Decision Tree Regression : 
#     train MedAE score : {medae_train}
#     test MedAE score : {medae_test}
#     train MAE Score : {mae_train}
#     test MAE Score : {mae_test}
# """)

# # %% [markdown]
# # ### KNeighbor Regressor

# # %%
# KNeighborsModel = Pipeline([
#     ('preprocessor', preprocessornd),
#     ('regressor', KNeighborsRegressor())
# ])

# # Fit the pipeline on the training data
# KNeighborsModel.fit(X_train, y_train)

# # %% [markdown]
# # decision tree regressor adalah non parametric supervised learning algorithm untuk regresi. bekerja dengan memprediksi target value dengan average dari value neighbor terdekat. jumlah dari neighbor (K) adalah parameter crucial dalam memutuskan model performance. algorithm mengukur jarak antara data point untuk menemukan neighbor terdekat

# # %%
# y_pred_tr_kn = KNeighborsModel.predict(X_train)
# y_pred_ts_kn = KNeighborsModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_kn)
# medae_test = median_absolute_error(y_test, y_pred_ts_kn)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_kn)
# mae_test = mean_absolute_error(y_test, y_pred_ts_kn)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_kn)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_kn)

# row = {
#     'model_name': 'K Neighbor Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     K Neighbor Regression : 
#     train MedAE score : {medae_train}
#     test MedAE score : {medae_test}
#     train MAE Score : {mae_train}
#     test MAE Score : {mae_test}
# """)

# # %% [markdown]
# # ### Tuned KNeigbor Regressor

# # %%
# TunedKNeighborModel = Pipeline([
#     ('preprocessor', preprocessornd),
#     ('regressor', KNeighborsRegressor())
# ])

# param_KN = {
#     'regressor__n_neighbors': range(1, 150, 2), 
#     'regressor__weights': ['uniform', 'distance'], 
#     'regressor__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 
#     'regressor__p': [1, 2]
# }

# scoring = { 'Median_Absolute_Error': 'neg_median_absolute_error', 'Mean_Absolute_Error': 'neg_mean_absolute_error' }

# GSCV_KN = GridSearchCV(TunedKNeighborModel, param_KN, cv=skf, scoring=scoring, refit='Median_Absolute_Error', n_jobs=12)
# GSCV_KN.fit(X_train, y_train)

# # %% [markdown]
# # tuning parameter pada KNeighbor,
# # - n_neighbors : jumlah neighbor yang digunakan
# # - weights : weight yang digunakan untuk prediksi
# #     - uniform : semua poin memiliki berat yang sama. 
# #     - distance : point memiliki berat yang berbeda tergantung pada beratnya. (semakin dekat semakin berpengaruh)
# # - algorithm : algorithm yang digunakan untuk menghitung neighbor terdekat, 
# #     - auto : secara otomatis mencoba memutuskan algoritma yang paling tebpat berdasarkan nilai yang diberikan ke fit method.
# #     - ball_tree : menggunakan sklearn.neighbors.BallTree algorithm
# #     - kd_tree : menggunakan sklearn. neighbors.KDTree algorithm
# #     - brute : menggunakan bruteforce search.  
# # - p : power parameter untuk Minkowski metric, 
# #     - 1 : manhattan_distance
# #     - 2 : euclidean_distance

# # %%
# GSCV_KN.best_score_

# # %%
# pd.DataFrame(GSCV_KN.cv_results_)[pd.DataFrame(GSCV_KN.cv_results_)['rank_test_Median_Absolute_Error'] == 1][['params', 'mean_test_Median_Absolute_Error']]

# # %%
# best_params = {k.split('__')[1]: v for k, v in GSCV_KN.best_params_.items() if k.split('__')[1] in ['n_neighbors', 'weights', 'algorithm', 'p']}

# BestKNeighborModel = Pipeline([
#     ('preprocessor', preprocessornd),
#     ('regressor', KNeighborsRegressor(**best_params))
# ])

# # Fit the pipeline on the training data
# BestKNeighborModel.fit(X_train, y_train)

# # %%
# y_pred_tr_kn_b = BestKNeighborModel.predict(X_train)
# y_pred_ts_kn_b = BestKNeighborModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_kn_b)
# medae_test = median_absolute_error(y_test, y_pred_ts_kn_b)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_kn_b)
# mae_test = mean_absolute_error(y_test, y_pred_ts_kn_b)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_kn_b)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_kn_b)

# row = {
#     'model_name': 'Tuned KNeighbor Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     Tuned KNeighbor Regression : 
#     train MedAE score : {medae_train}
#     test MedAE score : {medae_test}
#     train MAE Score : {mae_train}
#     test MAE Score : {mae_test}
# """)

# # %% [markdown]
# # ### Ensemble - Random Forest Regressor

# # %%
# RandomForestModel = Pipeline([
#     ('preprocessor', preprocessornd),
#     ('regressor', RandomForestRegressor())
# ])

# # Fit the pipeline on the training data
# RandomForestModel.fit(X_train, y_train)

# # %% [markdown]
# # random forest regressor model adalah learning methode  ensembles yang membangun beberapa decistion tree pada saat training dan menghasilkan rata-rata prediksi dari setiap tree untuk nilai regresinya. metode ini menggunakan konsep bagging(boostrap aggregating) untuk membuat beragam subset dari dataset asli dan menyesuaikan setiap subset dari dataset asli dan menyesuaikan setiap subset ke decision tree, mengurangi resiki overfitting dan meningkatkan generalisasi. tehnique ini robust, dapat digunakan pada large dataset, dan memberikan akurasi tinggi dengan menangkap pattern complex dari data. forest regressor digunakan secara luat karena fleksibilitasnya dan kemampuannya untuk menangani fitur numerik dan kategoris secara efektif

# # %%
# y_pred_tr_rf = RandomForestModel.predict(X_train)
# y_pred_ts_rf = RandomForestModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_rf)
# medae_test = median_absolute_error(y_test, y_pred_ts_rf)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_rf)
# mae_test = mean_absolute_error(y_test, y_pred_ts_rf)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_rf)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_rf)

# row = {
#     'model_name': 'Random Forest Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }
# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     Random Forest Regression : 
#     train MedAE score : {medae_train}
#     test MedAE score : {medae_test}
#     train MAE Score : {mae_train}
#     test MAE Score : {mae_test}
# """)

# # %% [markdown]
# # ### Tuned Ensemble - Random Forest Regressor

# # %%
# TunedRandomForestModel = Pipeline([
#     ('preprocessor', preprocessornd),
#     ('regressor', RandomForestRegressor())
# ])

# param_RF = {
#     'regressor__n_estimators': range(1, 200, 2), 
#     'regressor__max_depth': list(range(1, 51, 2)) + [None], 
#     'regressor__min_samples_split': range(2, 21), 
#     'regressor__min_samples_leaf': range(1, 21), 
#     'regressor__max_features': [1.0, 'sqrt', 'log2']
# }

# scoring = { 'Median_Absolute_Error': 'neg_median_absolute_error', 'Mean_Absolute_Error': 'neg_mean_absolute_error' }

# RSCV_RF = RandomizedSearchCV(TunedRandomForestModel, param_RF, cv=skf, scoring=scoring, refit='Median_Absolute_Error', n_jobs=12, error_score='raise')
# RSCV_RF.fit(X_train, y_train)

# # %% [markdown]
# # tuning parameter pada random forest,
# # - n_estimators : jumlah tree pada forest
# # - max_depth : kedalaman(depth) maximum dari tree. 
# # - min_samples_split : jumlah minimum sample yang diperlukan untuk memisahkan sebuah internal node
# # - min_samples_leaf : jumlah minimum sample yang diperlukan untuk membuat sebuah leaf node
# # - max_features : jumlah feature yang di pertimbangkan setiap ketika mencari spilt terbaik
# #     - 1.0 : max_features = jumlah feature yang ada
# #     - sqrt : max_features = sqrt(jumlah feature)
# #     - log2 : max_features = log2(jumlah feature)

# # %%
# RSCV_RF.best_score_

# # %%
# pd.DataFrame(RSCV_RF.cv_results_)[pd.DataFrame(RSCV_RF.cv_results_)['rank_test_Median_Absolute_Error'] == 1][['params', 'mean_test_Median_Absolute_Error']]

# # %%
# best_params = {k.split('__')[1]: v for k, v in RSCV_RF.best_params_.items() if k.split('__')[1] in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']}

# BestRandomForestModel = Pipeline([
#     ('preprocessor', preprocessornd),
#     ('regressor', RandomForestRegressor(**best_params))
# ])

# # Fit the pipeline on the training data
# BestRandomForestModel.fit(X_train, y_train)

# # %%
# y_pred_tr_rf_b = BestRandomForestModel.predict(X_train)
# y_pred_ts_rf_b = BestRandomForestModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_rf_b)
# medae_test = median_absolute_error(y_test, y_pred_ts_rf_b)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_rf_b)
# mae_test = mean_absolute_error(y_test, y_pred_ts_rf_b)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_rf_b)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_rf_b)

# row = {
#     'model_name': 'Tuned Random Forest Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     Tuned Random Forest Regression : 
#     train MedAE score : {medae_train}
#     test MedAE score : {medae_test}
#     train MAE Score : {mae_train}
#     test MAE Score : {mae_test}
# """)

# # %% [markdown]
# # ### Ensemble - Gradient Boosting Regressor

# # %%
# GradientBoostingModel = Pipeline([
#     ('preprocessor', preprocessornd),
#     ('regressor', GradientBoostingRegressor())
# ])

# # Fit the pipeline on the training data
# GradientBoostingModel.fit(X_train, y_train)

# # %% [markdown]
# # gradient boosting regressor adalah machine learning algorithm yang powerful. bekerja dengan sekumpulan weak prediction model, biasanyaa decision tree, secara bertahap. dengan menambahkan tree secara sequential dimana setiap tree memperbaiki error pada tree sebelumnya, ini meminimalisir loff sunction dan menaikkan akurasi prediksi, process iterasi ini memperbolehkan model untuk menangkap complex patterns dan interaksi dari data. gradient boosting regressor sangat effective untuk predictive moddelling dan biasanya digunakan karena kemampuannya memeberikan high accuracy dan robustness terhadap overfitting. 

# # %%
# y_pred_tr_gb = GradientBoostingModel.predict(X_train)
# y_pred_ts_gb = GradientBoostingModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_gb)
# medae_test = median_absolute_error(y_test, y_pred_ts_gb)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_gb)
# mae_test = mean_absolute_error(y_test, y_pred_ts_gb)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_gb)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_gb)

# row = {
#     'model_name': 'Gradient Boosting Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     Gradient Boosting Regression : 
#     train MedAE score : {medae_train}
#     test MedAE score : {medae_test}
#     train MAE Score : {mae_train}
#     test MAE Score : {mae_test}
# """)

# # %% [markdown]
# # ### Tuned - Gradient Boosting Regressor

# # %%
# TunedGradientBoostingModel = Pipeline([
#     ('preprocessor', preprocessornd),
#     ('regressor', GradientBoostingRegressor(random_state=42))
# ])

# param_GB = {
#     'regressor__n_estimators': randint(50, 300), 
#     'regressor__learning_rate': uniform(0.01, 0.2), 
#     'regressor__max_depth': randint(3, 10), 
#     'regressor__min_samples_split': randint(2, 21), 
#     'regressor__min_samples_leaf': randint(1, 21), 
#     'regressor__subsample': uniform(0.8, 0.2), 
#     'regressor__max_features': ['log2', 'sqrt',]
# }

# scoring = { 'Median_Absolute_Error': 'neg_median_absolute_error', 'Mean_Absolute_Error': 'neg_mean_absolute_error' }

# RSCV_GB = RandomizedSearchCV(TunedGradientBoostingModel, param_GB, cv=skf, scoring=scoring, refit='Median_Absolute_Error', n_jobs=16)
# RSCV_GB.fit(X_train, y_train)

# # %% [markdown]
# # tuning parameter pada Gradient Boosting,
# # - n_estimators : jumlah boosting stages yang akan dilakukan
# # - learning_rate : learning rate mengecilkan kontribusi setiap pohon dengan learning_rate. ada trade-off antara learning rate dan n_estimator
# # - max_depth : kedalaman(depth) maximum dari tree. 
# # - min_samples_split : jumlah minimum sample yang diperlukan untuk memisahkan sebuah internal node
# # - min_samples_leaf : jumlah minimum sample yang diperlukan untuk membuat sebuah leaf node
# # - subsambple : fraction dari sample yang digunakan untuk fitting individual base learners. 
# # - max_features : jumlah feature yang di pertimbangkan setiap ketika mencari spilt terbaik
# #     - sqrt : max_features = sqrt(jumlah feature)
# #     - log2 : max_features = log2(jumlah feature)

# # %%
# RSCV_GB.best_score_

# # %%
# pd.DataFrame(RSCV_GB.cv_results_)[pd.DataFrame(RSCV_GB.cv_results_)['rank_test_Median_Absolute_Error'] == 1][['params', 'mean_test_Median_Absolute_Error']]

# # %%
# best_params = {k.split('__')[1]: v for k, v in RSCV_GB.best_params_.items() if k.split('__')[1] in ['n_neighbors', 'learning_rate', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'subsample', 'max_features']}

# BestTunedGradientBoostingModel = Pipeline([
#     ('preprocessor', preprocessornd),
#     ('regressor', GradientBoostingRegressor(**best_params))
# ])

# # Fit the pipeline on the training data
# BestTunedGradientBoostingModel.fit(X_train, y_train)

# # %%
# y_pred_tr_gb_b = BestTunedGradientBoostingModel.predict(X_train)
# y_pred_ts_gb_b = BestTunedGradientBoostingModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_gb_b)
# medae_test = median_absolute_error(y_test, y_pred_ts_gb_b)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_gb_b)
# mae_test = mean_absolute_error(y_test, y_pred_ts_gb_b)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_gb_b)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_gb_b)

# row = {
#     'model_name': 'Tuned Gradient Boosting Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     Tuned Gradient Boosting Regression : 
#     train MedAE score : {medae_train}
#     test MedAE score : {medae_test}
#     train MAE Score : {mae_train}
#     test MAE Score : {mae_test}
# """)

# %% [markdown]
# ### Ensemble - Stacking Regressor

# %%
# base_models = [
#     ('lr', LinearRegression()),
#     ('dt', DecisionTreeRegressor())
# ]
# meta_model = RandomForestRegressor(random_state=42)
# stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# StackingRegressorModel =  Pipeline([
#     ('preprocessor', preprocessor),
#     ('stacking_regressor', stacking_regressor)
# ])

# StackingRegressorModel.fit(X_train, y_train)

# # %% [markdown]
# # stacking regressor model adalah meta_learner yang mencombinasikan beberapa regression model untuk meningkatkan predictive performancenya. bekerja dengan men-train beberapa base regressor pada dataset yang sama, dan menggunkan prediksi mereka  sebagai input pada sebuah final estimator (meta_tregressor), yang kan membuat final predictionnya. methode ini menggunkan kekuatan dari beberapa algorithm, mengurangi overfitting dan meningkatkan aaccuracy dengan menggabungkan prediksi yang berbeda-beda. stacking regressor effecitve pada base models yang memiliki strenght yang mengcomplement satu sama lain, membuatnya menjadi tool yang versatile dan powerful untuk regresi. 

# # %%
# y_pred_tr_sr = StackingRegressorModel.predict(X_train)
# y_pred_ts_sr = StackingRegressorModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_sr)
# medae_test = median_absolute_error(y_test, y_pred_ts_sr)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_sr)
# mae_test = mean_absolute_error(y_test, y_pred_ts_sr)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_sr)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_sr)

# row = {
#     'model_name': 'Stacking Regressor Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     Stacking Regressor Regression : 
#     train MedAE score : {medae_train}
#     test MedAE score : {medae_test}
#     train MAE Score : {mae_train}
#     test MAE Score : {mae_test}
# """)

# %% [markdown]
# ### Tuned Ensemble - Stacking Regressor

# %%
# BestDecisionTreeModel
# BestRandomForestModel
# LinRegModel

# base_models = [
#         ('linear', LinearRegression()),
#         ('decision_tree', DecisionTreeRegressor(**{k: v for k, v in BestDecisionTreeModel.get_params().items() if k in DecisionTreeRegressor().get_params()})),
#         ('random_forest', RandomForestRegressor(**{k: v for k, v in BestRandomForestModel.get_params().items() if k in DecisionTreeRegressor().get_params()}))
# ]

# meta_model = RandomForestRegressor(n_estimators=100, random_state=42)

# stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# BestStackingRegressorModel =  Pipeline([
#     ('preprocessor', preprocessornd),
#     ('stacking_regressor', stacking_regressor)
# ])

# BestStackingRegressorModel.fit(X_train, y_train)

# # %% [markdown]
# # versi tuned dari stacking regressor, base model menggunakan parameter terbaik yang telah digunakan pada training sebelumnya

# # %%
# y_pred_tr_sr_b = BestStackingRegressorModel.predict(X_train)
# y_pred_ts_sr_b = BestStackingRegressorModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_sr_b)
# medae_test = median_absolute_error(y_test, y_pred_ts_sr_b)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_sr_b)
# mae_test = mean_absolute_error(y_test, y_pred_ts_sr_b)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_sr_b)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_sr_b)

# row = {
#     'model_name': 'Tuned Stacking Regressor Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }


# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     Tuned Stacking Regressor Regression : 
#     train MedAE score : {medae_train}
#     test MedAE score : {medae_test}
#     train MAE Score : {mae_train}
#     test MAE Score : {mae_test}
# """)

# %% [markdown]
# ### XGBoost

# %%
# XGBModel =  Pipeline([
#     ('preprocessor', preprocessor),
#     ('stacking_regressor', XGBRegressor())
# ])

# XGBModel.fit(X_train, y_train)

# # %% [markdown]
# # XGBoost adalah optimized distributed gradient boosting library yang di designed untuk mejadi sangat efficient, flexible, dan portable. menggunakan algorithm dari gradiet boosting framework. xgboost menyediakan paraller tree boosting yang dapat menyelesaikan banyak masalah data science dengan cepat dan accurate. 

# # %%
# y_pred_tr_xgb = XGBModel.predict(X_train)
# y_pred_ts_xgb = XGBModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_xgb)
# medae_test = median_absolute_error(y_test, y_pred_ts_xgb)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_xgb)
# mae_test = mean_absolute_error(y_test, y_pred_ts_xgb)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_xgb)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_xgb)

# row = {
#     'model_name': 'XGBoost Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     XGBoost Regression : 
#     train MedAE score : {medae_train}
#     test MedAE score : {medae_test}
#     train MAE Score : {mae_train}
#     test MAE Score : {mae_test}
# """)

# # %% [markdown]
# # ### Tuned XGBoost

# # %%
# TunedXGBModel = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regressor', XGBRegressor())
# ])

# skf = KFold(n_splits=5, random_state=42, shuffle=True)
# param_XGB = { 
#     'regressor__n_estimators': randint(100, 1001),
#     'regressor__max_depth': randint(3, 11),
#     'regressor__learning_rate': uniform(0.01, 0.3),
#     'regressor__min_child_weight': randint(1, 11),
#     'regressor__gamma': uniform(0, 1),
#     'regressor__reg_alpha': uniform(0, 1),
#     'regressor__reg_lambda': uniform(0, 1)
# }

# scoring = { 'Median_Absolute_Error': 'neg_median_absolute_error', 'Mean_Absolute_Error': 'neg_mean_absolute_error' }

# RSCV_XGB = RandomizedSearchCV(TunedXGBModel, param_XGB, cv=skf, scoring=scoring, refit='Median_Absolute_Error', n_jobs=12, error_score='raise')
# RSCV_XGB.fit(X_train, y_train)

# %% [markdown]
# tuning parameter XGBoost,
# - n_estimators : jumlah boosting stages yang akan dilakukan
# - max_depth : kedalaman(depth) maximum dari tree
# - learning_rate : learning rate mengecilkan kontribusi setiap pohon dengan learning_rate. ada trade-off antara learning rate dan n_estimator
# - min_child_weight : minium sum of iinstance weight yang dibutuhkan pada sebuah child
# - subsambple : fraction dari sample yang digunakan untuk fitting individual base learners. 
# - gamma : minium loss reduction yang dibutuhkan untuk membuat partisi selanjutnya pada sebuah leaf node dari tree
# - colsample_bytree : subsample ratio of columns ketika membuat setiap tree. subsampling terjadi setiap tree dibuat.
# - reg_alpha : L2 regularization term pada weights. menambahkan nilai ini membuat cmodel semakin conservative. 
# - reg_lambda : L1 regularization term pada weights. menambahkan nilai ini membuat cmodel semakin conservative. 

# %%
# RSCV_XGB.best_score_

# # %%
# pd.DataFrame(RSCV_XGB.cv_results_)[pd.DataFrame(RSCV_XGB.cv_results_)['rank_test_Median_Absolute_Error'] == 1][['params', 'mean_test_Median_Absolute_Error']]

# # %%
# best_params = {k.split('__')[1]: v for k, v in RSCV_XGB.best_params_.items() if k.split('__')[1] in ['max_depth', 'learning_rate', 'n_estimators', 'min_child_weight', 'subsample', 'gamma', 'colsample_bytree', 'reg_alpha', 'reg_lambda']}

# BestXGBModel = Pipeline([
#     ('preprocessor', preprocessornd),
#     ('regressor', XGBRegressor(**best_params))
# ])

# # Fit the pipeline on the training data
# BestXGBModel.fit(X_train, y_train)

# # %%
# y_pred_tr_xgb_b = BestXGBModel.predict(X_train)
# y_pred_ts_xgb_b = BestXGBModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_xgb_b)
# medae_test = median_absolute_error(y_test, y_pred_ts_xgb_b)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_xgb_b)
# mae_test = mean_absolute_error(y_test, y_pred_ts_xgb_b)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_xgb_b)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_xgb_b)

# row = {
#     'model_name': 'Tuned XGBoost Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }


# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     Tuned XGBoost Regression : 
#     train MedAE score : {medae_train}
#     test MedAE score : {medae_test}
#     train MAE Score : {mae_train}
#     test MAE Score : {mae_test}
# """)

# # %% [markdown]
# # ### LightGBM

# # %%
# LGBMModel = Pipeline([
#     ('preprocessor', preprocessornd),
#     ('stacking_regressor', LGBMRegressor())
# ])

# LGBMModel.fit(X_train, y_train)

# # %% [markdown]
# # LightGBM adlah sebuah gradient boosting framework yang menggunakan tree based learning algorithm, yang di design untuk di-distribusikan dan efficient. 
# # perbedaannya XGBoost,
# # 1. **Kecepatan dan Efisiensi**:
# # - **LightGBM**: Menggunakan algorithm berbasis histogram dan mendukung pertumbuhan pohon berdasarkan leaf-wise (best-first). Hal ini memungkinkan LightGBM menjadi lebih cepat dan lebih hemat memori, terutama dengan large dataset.
# # - **XGBoost**: Menggunakan pertumbuhan pohon depth-wise (kedalaman), yang dapat lebih lambat dan menghabiskan lebih banyak memori dibandingkan dengan LightGBM.
# # 
# # 2. **Penanganan largedataset**:
# # - **LightGBM**: Dapat menangani large dataset dengan lebih efisien karena pendekatan berbasis histogram dan penggunaan memori yang lebih baik.
# # - **XGBoost**: Meskipun juga mampu menangani large dataset, membutuhkan lebih banyak tuning dan resource untuk mencapai kinerja yang sama seperti LightGBM.
# # 
# # 3. **training Paralel dan GPU**:
# # - **LightGBM**: Mendukung training paralel dan GPU support, yang dapat lebih mempercepat waktu training. 
# # - **XGBoost**: Juga mendukung training paralel dan GPU, tetapi implementasi dalam LightGBM sering dianggap lebih efisien.
# # 
# # 4. **accuracy dan Performance**:
# # Baik LightGBM maupun XGBoost dikenal karena accuracy dan robustness yang tinggi. Pilihan di antara keduanya sering kali bergantung pada dataset dan problemnya. LightGBM mungkin memiliki keunggulan dalam hal kecepatan dan penanganan large dataset, sementara XGBoost dikenal karena fleksibilitasnya dan tuning parameter yang luas.
# # 
# # 5. **Strategi tree growth**:
# # - **LightGBM**: Menumbuhkan pohon dari daun ke daun, yang dapat menghasilkan pohon yang lebih dalam dan berpotensi lebih akurat.
# # - **XGBoost**: Menumbuhkan pohon dari tingkat ke tingkat, yang dapat lebih seimbang tetapi mungkin kurang efisien dalam beberapa kasus.

# # %%
# y_pred_tr_lgbm = LGBMModel.predict(X_train)
# y_pred_ts_lgbm = LGBMModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_lgbm)
# medae_test = median_absolute_error(y_test, y_pred_ts_lgbm)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_lgbm)
# mae_test = mean_absolute_error(y_test, y_pred_ts_lgbm)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_lgbm)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_lgbm)

# row = {
#     'model_name': 'LightBGM Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }


# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     LightBGM Regression : 
#     train MedAE score : {medae_train}
#     test MedAE score : {medae_test}
#     train MAE Score : {mae_train}
#     test MAE Score : {mae_test}
# """)

# %% [markdown]
# ### Tuned LightGBM

# %%
# TunedLGBMModel = Pipeline([
#     ('preprocessor', preprocessornd),
#     ('regressor', LGBMRegressor(random_state=42, force_row_wise=True))
# ])

# skf = KFold(n_splits=5, random_state=42, shuffle=True)
# param_LGBM = { 
#     'regressor__max_depth' : list(range(2, 150, 2)) + [None],
#     'regressor__num_leaves' : list(range(5, 32, 2)) + [None],
#     'regressor__learning_rate': np.arange(0.01, 0.21, 0.02),
#     'regressor__n_estimators': [50, 100, 300, 500],
#     'regressor__reg_alpha': np.arange(0, 1, 0.2),
#     'regressor__reg_lambda': np.arange(0, 1, 0.2)
# }

# scoring = { 'Median_Absolute_Error': 'neg_median_absolute_error', 'Mean_Absolute_Error': 'neg_mean_absolute_error' }

# RSCV_LGBM = RandomizedSearchCV(TunedLGBMModel, param_LGBM, cv=skf, scoring=scoring, refit='Median_Absolute_Error', n_jobs=12)
# RSCV_LGBM.fit(X_train, y_train)

# # %%
# RSCV_LGBM.best_score_

# # %%
# RSCV_LGBM.best_score_

# # %%
# pd.DataFrame(RSCV_LGBM.cv_results_)[pd.DataFrame(RSCV_LGBM.cv_results_)['rank_test_Median_Absolute_Error'] == 1][['params', 'mean_test_Median_Absolute_Error']]

# # %%
# best_params = {k.split('__')[1]: v for k, v in RSCV_LGBM.best_params_.items() if k.split('__')[1] in ['max_depth', 'num_leaves', 'learning_rate', 'n_estimators', 'reg_alpha', 'reg_lambda']}

# BestLGBMModel = Pipeline([
#     ('preprocessor', preprocessornd),
#     ('regressor', LGBMRegressor(**best_params,  force_row_wise=True))
# ])

# # Fit the pipeline on the training data
# BestLGBMModel.fit(X_train, y_train)

# # %%
# y_pred_tr_lgbm_b = BestLGBMModel.predict(X_train)
# y_pred_ts_lgbm_b = BestLGBMModel.predict(X_test)

# medae_train = median_absolute_error(y_train, y_pred_tr_lgbm_b)
# medae_test = median_absolute_error(y_test, y_pred_ts_lgbm_b)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr_lgbm_b)
# mae_test = mean_absolute_error(y_test, y_pred_ts_lgbm_b)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr_lgbm_b)
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts_lgbm_b)

# row = {
#     'model_name': 'Tuned LightBGM Regression',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }


# # scores = pd.concat([scores, pd.DataFrame(row)], ignore_index=True)
# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# print(f"""
#     Tuned LightBGM Regression : 
#     train MedAE score : {medae_train}
#     test MedAE score : {medae_test}
#     train MAE Score : {mae_train}
#     test MAE Score : {mae_test}
# """)

# %% [markdown]
# ### Scores

# %%
# scores.sort_values(by=['MedAE_Test'])

# %% [markdown]
# ### Type sebagai pengganti Make

# %% [markdown]
# Type akan dicoba sebagai pengganti Make, untuk menyingkat waktu, machine learning hanya akan dilakukan pada top 5 model + 1 model terstabil yaitu, 
# - Tuned XGBoost Regressor
# - Random Forest Regressor
# - XGBoost Regressor
# - Tuned LightBGM Regressor
# - Tuned Stacking Regressor 
# - Tuned Gradient Boosting

# %% [markdown]
# #### Data Splitting

# %%
X_Type_train, X_Type_test, y_train, y_test = train_test_split(X_Type, y, test_size = .2, random_state=42)

# %% [markdown]
# #### Preprocessor

# %%
preprocessorType = ColumnTransformer([
    ('OE', OrdinalEncoder(categories=categories), ['Options']),
    ('OH', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), ['Type', 'Origin', 'Color', 'Fuel_Type', 'Gear_Type', 'Region']),
    ('MMS', MinMaxScaler(), ['Year', 'Engine_Size']),
    ('RS', RobustScaler(), ['Mileage']),
])

# %%
preprocessorndType = ColumnTransformer([
    ('OE', OrdinalEncoder(categories=categories), ['Options']),
    ('OH', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'), ['Type', 'Origin', 'Color', 'Fuel_Type', 'Gear_Type', 'Region']),
    ('MMS', MinMaxScaler(), ['Year', 'Engine_Size']),
    ('RS', RobustScaler(), ['Mileage']),
])

# %% [markdown]
# #### Random Forest

# %%
# TypeRandomForestModel = Pipeline([
#     ('preprocessor', preprocessorndType),
#     ('regressor', RandomForestRegressor())
# ])

# TypeRandomForestModel.fit(X_Type_train, y_train)

# y_pred_tr = TypeRandomForestModel.predict(X_Type_train)
# y_pred_ts = TypeRandomForestModel.predict(X_Type_test)

# medae_train = median_absolute_error(y_train, y_pred_tr)
# medae_test = median_absolute_error(y_test, y_pred_ts)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr)
# mae_test = mean_absolute_error(y_test, y_pred_ts)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr) 
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts)

# row = {
#     'model_name': 'Random Forest Regression (Type)',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# %% [markdown]
# #### XGBoost

# %%
TypeXGBModel = Pipeline([
    ('preprocessor', preprocessorndType),
    ('regressor', XGBRegressor())
])

TypeXGBModel.fit(X_Type_train, y_train)

y_pred_tr = TypeXGBModel.predict(X_Type_train)
y_pred_ts = TypeXGBModel.predict(X_Type_test)

medae_train = median_absolute_error(y_train, y_pred_tr)
medae_test = median_absolute_error(y_test, y_pred_ts)
medae_diff = medae_test - medae_train
mae_train = mean_absolute_error(y_train, y_pred_tr)
mae_test = mean_absolute_error(y_test, y_pred_ts)
mae_diff = mae_test - mae_train
mape_train = mean_absolute_percentage_error(y_train, y_pred_tr) 
mape_test = mean_absolute_percentage_error(y_test, y_pred_ts)

row = {
    'model_name': 'XGBoost Regression (Type)',
    'MedAE_train': medae_train,
    'MedAE_Test': medae_test,
    'MedAE_diff': medae_diff,
    'MAE_train': mae_train,
    'MAE_Test': mae_test,
    'MAE_diff': mae_diff,
    'MAPE_train': mape_train,
    'MAPE_Test': mape_test
}

scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# %% [markdown]
# #### Tuned XGBoost

# %%
TunedTypeXGBModel = Pipeline([
    ('preprocessor', preprocessorndType),
    ('regressor', XGBRegressor())
])

skf = KFold(n_splits=5, random_state=42, shuffle=True)
param_XGB = { 
    'regressor__n_estimators': randint(100, 1001),
    'regressor__max_depth': randint(3, 11),
    'regressor__learning_rate': uniform(0.01, 0.3),
    'regressor__min_child_weight': randint(1, 11),
    'regressor__gamma': uniform(0, 1),
    'regressor__reg_alpha': uniform(0, 1),
    'regressor__reg_lambda': uniform(0, 1)
}

scoring = { 'Median_Absolute_Error': 'neg_median_absolute_error', 'Mean_Absolute_Error': 'neg_mean_absolute_error' }

RSCV_T_XGB = RandomizedSearchCV(TunedTypeXGBModel, param_XGB, cv=skf, scoring=scoring, refit='Median_Absolute_Error', n_jobs=12, error_score='raise')
RSCV_T_XGB.fit(X_Type_train, y_train)

best_params = {k.split('__')[1]: v for k, v in RSCV_T_XGB.best_params_.items() if k.split('__')[1] in ['max_depth', 'learning_rate', 'n_estimators', 'min_child_weight', 'subsample', 'gamma', 'colsample_bytree', 'reg_alpha', 'reg_lambda']}

BestTypeXGBModel = Pipeline([
    ('preprocessor', preprocessorndType),
    ('regressor', XGBRegressor(**best_params))
])

# Fit the pipeline on the training data
BestTypeXGBModel.fit(X_Type_train, y_train)

y_pred_tr = BestTypeXGBModel.predict(X_Type_train)
y_pred_ts = BestTypeXGBModel.predict(X_Type_test)

medae_train = median_absolute_error(y_train, y_pred_tr)
medae_test = median_absolute_error(y_test, y_pred_ts)
medae_diff = medae_test - medae_train
mae_train = mean_absolute_error(y_train, y_pred_tr)
mae_test = mean_absolute_error(y_test, y_pred_ts)
mae_diff = mae_test - mae_train
mape_train = mean_absolute_percentage_error(y_train, y_pred_tr) 
mape_test = mean_absolute_percentage_error(y_test, y_pred_ts)

row = {
    'model_name': 'Tuned XGBoost Regression (Type)',
    'MedAE_train': medae_train,
    'MedAE_Test': medae_test,
    'MedAE_diff': medae_diff,
    'MAE_train': mae_train,
    'MAE_Test': mae_test,
    'MAE_diff': mae_diff,
    'MAPE_train': mape_train,
    'MAPE_Test': mape_test
}

scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# %% [markdown]
# #### Tuned Stacking Regressor

# %%
# BestDecisionTreeModel
# BestRandomForestModel
# LinRegModel

# base_models = [
#         ('linear', LinearRegression()),
#         ('decision_tree', DecisionTreeRegressor(**{k: v for k, v in BestDecisionTreeModel.get_params().items() if k in DecisionTreeRegressor().get_params()})),
#         ('random_forest', RandomForestRegressor(**{k: v for k, v in BestRandomForestModel.get_params().items() if k in DecisionTreeRegressor().get_params()}))
# ]

# meta_model = RandomForestRegressor(n_estimators=100, random_state=42)

# stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# BestTypeStackingRegressorModel =  Pipeline([
#     ('preprocessor', preprocessorndType),
#     ('stacking_regressor', stacking_regressor)
# ])

# BestTypeStackingRegressorModel.fit(X_Type_train, y_train)

# y_pred_tr = BestTypeStackingRegressorModel.predict(X_Type_train)
# y_pred_ts = BestTypeStackingRegressorModel.predict(X_Type_test)

# medae_train = median_absolute_error(y_train, y_pred_tr)
# medae_test = median_absolute_error(y_test, y_pred_ts)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr)
# mae_test = mean_absolute_error(y_test, y_pred_ts)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr) 
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts)

# row = {
#     'model_name': 'Tuned Stacking Regression (Type)',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# # %% [markdown]
# # #### Tuned LightBGM

# # %%
# TunedTypeLGBMModel = Pipeline([
#     ('preprocessor', preprocessorndType),
#     ('regressor', LGBMRegressor(random_state=42, force_row_wise=True))
# ])

# skf = KFold(n_splits=5, random_state=42, shuffle=True)
# param_LGBM = { 
#     'regressor__max_depth' : list(range(2, 150, 2)) + [None],
#     'regressor__num_leaves' : list(range(5, 32, 2)) + [None],
#     'regressor__learning_rate': np.arange(0.01, 0.21, 0.02),
#     'regressor__n_estimators': [50, 100, 300, 500],
#     'regressor__reg_alpha': np.arange(0, 1, 0.2),
#     'regressor__reg_lambda': np.arange(0, 1, 0.2)
# }

# scoring = { 'Median_Absolute_Error': 'neg_median_absolute_error', 'Mean_Absolute_Error': 'neg_mean_absolute_error' }

# RSCV_T_LGBM = RandomizedSearchCV(TunedTypeLGBMModel, param_LGBM, cv=skf, scoring=scoring, refit='Median_Absolute_Error', n_jobs=12)
# RSCV_T_LGBM.fit(X_Type_train, y_train)

# best_params = {k.split('__')[1]: v for k, v in RSCV_T_LGBM.best_params_.items() if k.split('__')[1] in ['max_depth', 'num_leaves', 'learning_rate', 'n_estimators', 'reg_alpha', 'reg_lambda']}

# BestTypeLGBMModel = Pipeline([
#     ('preprocessor', preprocessorndType),
#     ('regressor', LGBMRegressor(**best_params,  force_row_wise=True))
# ])

# # Fit the pipeline on the training data
# BestTypeLGBMModel.fit(X_Type_train, y_train)

# y_pred_tr = BestTypeLGBMModel.predict(X_Type_train)
# y_pred_ts = BestTypeLGBMModel.predict(X_Type_test)

# medae_train = median_absolute_error(y_train, y_pred_tr)
# medae_test = median_absolute_error(y_test, y_pred_ts)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr)
# mae_test = mean_absolute_error(y_test, y_pred_ts)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr) 
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts)

# row = {
#     'model_name': 'LightBGM Regression (Type)',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# %% [markdown]
# #### Tuned Gradient Boosting

# %%
# TunedTypeGradientBoostingModel = Pipeline([
#     ('preprocessor', preprocessorndType),
#     ('regressor', GradientBoostingRegressor(random_state=42))
# ])

# param_GB = {
#     'regressor__n_estimators': randint(50, 300), 
#     'regressor__learning_rate': uniform(0.01, 0.2), 
#     'regressor__max_depth': randint(3, 10), 
#     'regressor__min_samples_split': randint(2, 21), 
#     'regressor__min_samples_leaf': randint(1, 21), 
#     'regressor__subsample': uniform(0.8, 0.2), 
#     'regressor__max_features': ['log2', 'sqrt',]
# }

# scoring = { 'Median_Absolute_Error': 'neg_median_absolute_error', 'Mean_Absolute_Error': 'neg_mean_absolute_error' }

# RSCV_T_GB = RandomizedSearchCV(TunedTypeGradientBoostingModel, param_GB, cv=skf, scoring=scoring, refit='Median_Absolute_Error', n_jobs=16)
# RSCV_T_GB.fit(X_Type_train, y_train)

# best_params = {k.split('__')[1]: v for k, v in RSCV_T_GB.best_params_.items() if k.split('__')[1] in ['n_neighbors', 'learning_rate', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'subsample', 'max_features']}

# BestTypeGradientBoostingModel = Pipeline([
#     ('preprocessor', preprocessorndType),
#     ('regressor', GradientBoostingRegressor(**best_params))
# ])

# # Fit the pipeline on the training data
# BestTypeGradientBoostingModel.fit(X_Type_train, y_train)

# y_pred_tr = BestTypeGradientBoostingModel.predict(X_Type_train)
# y_pred_ts = BestTypeGradientBoostingModel.predict(X_Type_test)

# medae_train = median_absolute_error(y_train, y_pred_tr)
# medae_test = median_absolute_error(y_test, y_pred_ts)
# medae_diff = medae_test - medae_train
# mae_train = mean_absolute_error(y_train, y_pred_tr)
# mae_test = mean_absolute_error(y_test, y_pred_ts)
# mae_diff = mae_test - mae_train
# mape_train = mean_absolute_percentage_error(y_train, y_pred_tr) 
# mape_test = mean_absolute_percentage_error(y_test, y_pred_ts)

# row = {
#     'model_name': 'Gradient Boosting Regression (Type)',
#     'MedAE_train': medae_train,
#     'MedAE_Test': medae_test,
#     'MedAE_diff': medae_diff,
#     'MAE_train': mae_train,
#     'MAE_Test': mae_test,
#     'MAE_diff': mae_diff,
#     'MAPE_train': mape_train,
#     'MAPE_Test': mape_test
# }

# scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True)

# %% [markdown]
# ### Scores (with Type)

# %%
# Convert MAPE values to percentage and format to 2 decimal places
scores['MAPE_train'] = scores['MAPE_train'] * 100
scores['MAPE_Test'] = scores['MAPE_Test'] * 100
                                                

# %%
scores.sort_values(by=['MedAE_Test']).round(2)

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ### Machine Learning Summary

# %% [markdown]
# Terbaik merupakan XGBoost Regression menggunakan Type dan bukan Make. dengan score
# | model_name | MedAE_train | MedAE_Test | MedAE_diff | MAE_train | MAE_Test | MAE_diff | MAPE_train | MAPE_Test |
# |---|---|---|---|---|---|---|---|---|
# |Tuned XGBoost Regression (Type)|3057.46|7429.29|4371.83|4385.11|14227.03|9841.93|8.04|20.87|
# |XGBoost Regression (Type)|7062.09|8672.35|1610.26|10110.83|15454.51|5343.67|17.32|22.79

# %% [markdown]
# - Nilai MAPE kurang dari 10% berarti model sangat akurat
# - Nilai MAPE antara 10%–20% berarti model baik
# - Nilai MAPE antara 20%–50% berarti model layak
# - Nilai MAPE lebih dari 50% berarti model buruk

# %% [markdown]
# model Tuned XGBoost Regression (Type) dengan score MAPE_train bernilai 8.04% dan MAPE_Test bernilai 20.87% ini berarti ada perbedaan sebesar 12,83%. ini berarti mobil kemungkinan mengalami sedikit overfit. model yang lebih stabil adalah model XGBoost Regression (Type) dengan  MAPE_train bernilai 17.32% dan MAPE_Test bernilai 22.79%.    
#    
# kedua model tersebut berhubungan (tuned dan base), jadi ada kemungkinan jika menggunakan gridsearchcv daripada randomsearccv untuk tuningnya, maka dapat menghasilkan model yang lebih baik. tetapi tidak gridsearch dengan tuning parameter tersebut tidak dapat dijalankan, karena keterbatasan hardware, dalam hal ini ram. 

# %% [markdown]
# menurut hasil model yang dibuat jika perkiraan harga adalah 60.000 maka range harganya adalah 45.772,97 - 74.227,03. yaitu dengan errorate sebesar 20%
# 

# %%
y_final_pred = BestTypeXGBModel.predict(X_Type_test)
error = mean_absolute_error(y_test, y_final_pred)

plt.figure(figsize=(14, 5))
plot = sns.scatterplot(x=y_test, y=y_final_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'g--')  # Add the line x=y
plt.fill_between([y_test.min(), y_test.max()], [y_test.min() - error, y_test.max() - error], [y_test.min() + error, y_test.max() + error], color='red', alpha=0.3)  # Add the safe zone
plot.set(title='Actual vs. Predicted Price', xlabel='Actual Price', ylabel='Predicted Price')
plt.grid()
plt.show()

# %% [markdown]
# dari graph ini terlihat bahwa prediksi range harga masih merangkum sebagian besar dari harga actual, terutama dibawah 300,000. data point masih lumayan dekat dengan garis linear x=y. yang berarti alignment yang lumayan baik antara value prediksi dan actual.  

# %% [markdown]
# 
# #### Recomendation
# 
# penambahan row dataset dan penambahan column yang lebih informatif seperti, column :
# - kesehatan ban
# - services history dengan tempatnya
# - interior and exterior condition
# - modification

# %% [markdown]
# ### Feature Importance

# %%
# ohe_feature_names = preprocessornd.transformers_[1][1].get_feature_names_out(preprocessor.transformers_[1][2]).tolist()
# ordinal_features = preprocessornd.transformers_[0][2]
# minmax_features = preprocessornd.transformers_[2][2]
# robust_features = preprocessornd.transformers_[3][2]

# # Combine all feature names
# feature_names = ordinal_features + ohe_feature_names + minmax_features + robust_features

# # %%
# pd.DataFrame({
#     'Feature' : feature_names,
#     'Importance %' : ((DecisionTreeModel.steps[1][1].feature_importances_) * 100).round(4),
# }).sort_values(by='Importance %', ascending=False).head(5)

# # %%
# pd.DataFrame({
#     'Feature' : feature_names,
#     'Importance %' : ((DecisionTreeModel.steps[1][1].feature_importances_) * 100).round(4),
# }).sort_values(by='Importance %').head(20)

# %% [markdown]
# 5 feature terpenting adalah Engine_Size, Year, Options, Mileage dan apakah brandnya adalah merceces. 
# mobil yang memiliki feature-feature ini lebih berpengaruh terhadap harga jualnya. 
# 
# sedangkan, feature-feature yang terendah adalah, 
# Make_FAW, Make_Chery, Make_Great Wall, Make_GAC, Make_Hummer, Make_Iveco, Make_Maserati, Make_Lifan, Make_Mercury, Make_Peugeot, Make_Zhengzhou, Make_Victory Auto, Region_Wadi Dawasir, Color_Green. 
# dimana feature-feature ini memeriki pengaruh yang sanget kecil terhadap harga jualnya. 

# %% [markdown]
# **rekomendasi**: 
# - Highlight Engine Size dan Year pada setiap listing, ini akan menarik buyer yang interested pada feature-feature tersebut. 
# - menawarkan custom filter pada pencarian Options dan Make: ini akan mempermudah user mencari mobil bekas dengan feature atau Brand yang dicari.
# - menawarkan custom sort function untuk Men-sort listing dengan Mileage dan Year: ini akan mempermudah user untuk melihat mobil yang lebih valuable (newer year dan low mileage)

# %% [markdown]
# ## Summary

# %% [markdown]
# - Accurately Predict the price of used cars.    
#     model yang telah dibuat dapat memprediksi harga mobil bekas dengan cukup akurat dengan MAPE sebesar 20,87. model menggunakan menggunakan XGBoost Regression yang telah di-tune. 
# - Understanding Key Factors. 
#     informasi yang didapatkan dari Feature Importances, 5 feature yang paling berpengaruh adalah, 
#     1. Engine_Size
#     2. Year
#     3. Options
#     4. Make : Mercedes
#     5. Mileage
#     dan yang palinng tidak berpengaruh ada 19 yaitu, 
#     1. Make : BYD
#     2. Make : Cherry
#     3. Make : FAW
#     4. Make : Iveco
#     5. Make : Hummer
#     6. Make : Great Wall
#     7. Make : GAC
#     8. Make : Foton
#     9. Make: Mercury
#     10.	Make : Subaru
#     11. Make : Zhengzhou
#     12.	Make : Škoda
#     13.	Make : Maserati
#     14.	Make : Peugeot
#     15.	Make : Lifan
#     16.	Color : Orange
#     17.	Region : Sakaka
#     18.	Region : Wadi Dawasir
#     19.	Region : Arar

# %% [markdown]
# pendapatan dari komisi penjualan mobil bekasi biasanya berada pada 25% dari profit penjualan mobil dengan minimun 125 usd[^1]. feature Subscription ini dapat menambahkan projected maximum profit 7,603[^2] * 150sar(40usd[^3]) yaitu 1.140.450 sar per bulannya sebagai sumber revenue baru. perusahaan juga dapat menambah sumber revenue baru yaitu dengan memberikan badge pada listing yang sudah di check oleh perusahaan. 
# 
# [^1]:https://blog.osum.com/car-salesman-commission/#:~:text=According%20to%20Motor%20Trend%2C%20the%20average%20commission%20for,based%20on%20the%20specific%20dealership%20and%20sales%20performance.
# [^2]:https://www.businesstimes.com.sg/singapore/used-car-sales-down-77-first-two-months-2023-new-car-registration-falls-238
# [^3]:https://sell.amazon.com/pricing
# 

# %% [markdown]
# ## Recomendation

# %% [markdown]
# dari Data Analysis : 
# - price distribution : mensegementasikan Price mejadi 3 yaitu, Budget(<60000), midrange(60000-185000) dan Highend(>185000)
# - brand/Type popularity : membuat bagian khusus dan penawaran untuk brand populer pada platform tergantung pada access region
# - Gear Popularity : menampilkan gear type tertentu tergantung pada harga yang dicari customer berdasarkan price_average-nya
# - Fuel Popularity : Memberikan Badge/flair/label pada mobil sesuai dengan fuel typenya. 
# - Mileage and Age : menhighlight mobil dengan mileage rendah jika age nya tinggi/tua terutama dengan harga dibawah price_average-nya. 
# 
# untuk dataset : 
# - menambahkan dataset populasi per region/region category
# - menambahkan dataset history pencarian/query pada platform
# - menambahkan column kesehatan ban
# - menambahkan column services beserta tempat servicesnya
# - menambahkan colum interior dan exterior services
# - menambahkan column modification
# 
# keuntungan perusahaan : 
# - menaikan revenue dengan meninggikan customer satisfaction
# - menambahkan feature yang berguna untuk menambah/mempercepat transaksi pada platform
# - pemasaran tertentu dapat menjangkau lebih banyak potential customer, yang dapat menaikan conversion
# - perusahaan juga dapat menjual feature "specified region advertisement", untuk seller yang mau memasang ads/promoted untuk mobilnya pada suatu region tertentu. 
# - customer marketing strategies dapat menaikkan platform brand dan customer loyalty dan membantu customer retention. 
# - 5% increase pada customer retention dapat menjadi 25%-95% increase pada profit, existing buyer mengeluarkan upto 300%, memakan 5x lipab lebih banyak uang untuk mendapatkan customer baru daripada me-retain yang sudah ada[^1]
# - dengan menambahkan datataset sesuai dengan yang telah di tuliskan, perusahaan dapat menawarkan sumber revenue baru berupa services kepada seller, dimana mobil mereka yang akan dijual dapat di test terlebih dahulu oleh perusahaan. dan jika lolos test sesuai criterianya, maka platform/perusahaan akan memberikan badge seperti "trusted seller" atau "tested" untuk menambahkan kepercayaan calon pembeli terhadap mobil yang akan dibeli


