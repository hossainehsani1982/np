import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


dfs = []
combined_dataset_path = r'E:\NpowerLab\DataSets\Project\combined_dataset.csv'

if os.path.isfile(combined_dataset_path):
    pass
else:
    for fileName in os.listdir(r'E:\NpowerLab\DataSets\Project'):
         if fileName.endswith('.csv'):
            df = pd.read_csv(os.path.join(r'E:\NpowerLab\DataSets\Project', fileName))
            df = df.iloc[:, :12]
            dfs.append(df)
            
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(combined_dataset_path, index=False)


df = pd.read_csv(combined_dataset_path)

'''
Data Cleaning
''' 

#remove rows without date
df = df.dropna(subset=['Date'])

#remove rows with invalid date
date_pattern = r"\d{4}-\d{2}-\d{2}"
df = df[df['Date'].str.match(date_pattern)]

#remove rows with nan values
nan_columns = df.columns[df.isna().any()].tolist()
df.drop(nan_columns, axis=1, inplace=True)

'''
Data Transformation
'''

#convert date to datetime
df['Date'] = pd.to_datetime(df['Date'])
#replace date with year
df['Date'] = df['Date'].dt.year
#rename date column to year
df.rename(columns={'Date': 'Year'}, inplace=True)

#group by year anf symbol
df = df.groupby(['Year', 'Symbol']).sum().reset_index()

#calculate the total volume of stock traded for each year based on the open and close AVRG price
open_colose_mean = df[['Open', 'Close']].mean(axis = 1)
df['Total'] = df['Volume'] * open_colose_mean

print(df.head())

####plotting
#area plot of the stock price of the companies over the years

df = df.pivot(index='Year', columns='Symbol', values='Total')
ax = df.plot.area()
years = df.index
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45)

formatter = plt.FuncFormatter(lambda x, _: "${:,.0f}".format(x))
ax.yaxis.set_major_formatter(formatter)


plt.title('Stock Price of Companies Over the Years')
plt.ylabel('Stock Price')
plt.xlabel('Years')
plt.legend(title='Companies', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


