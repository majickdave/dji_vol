import pandas as pd

df = pd.read_csv('full dataset.csv')
df.rename(columns={'ACTIVITY DATE':'date', 'Sum of TOTAL TIME - AHT (sec)': 'handle_time',
                  'Sum of TOTAL TIME - AHT - FORECAST (sec)': 'handle_time_forecast', 
                  'Sum of VOLUME - RECEIVED': 'volume', 'Sum of VOLUME - FORECAST': 'volume_forecast' }, 
          inplace=True)

# make all col names lower case
df.columns = [x.lower() for x in df.columns]
print(df.head())

for line in df['business line'].unique():
    data = df[df['business line'] == line][['date','handle_time','handle_time_forecast', 
                                            'volume', 'volume_forecast']]
    s = '../data/'+line+'_aht_vol.csv'
    s = s.replace(' ', '-')
    data.to_csv(s, index=False)