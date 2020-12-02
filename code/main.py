import os
from run_model import *
from split_dataset import *

def main():
    # load a dataframe
    df = load_rename('./data/full_dataset.csv')

    # create csv for individual business unit
    write_data(df)
    for k in ['handle_time', 'volume', 'aht']:
        end = ['2020-11-01']*10
        # start = ['2016-03-10','2016-01-20','2017-01-01','2016-04-30','2017-01-20','2017-01-10',
        # '2016-03-10','2016-04-30', '2017-01-01', '2017-01-01']
        start = ['2019-11-01'] *10
        kpi = [k]*10
        files = os.listdir('./data/')

        files = list(filter(lambda x: (x[-7:] == 'vol.csv'), files))

        for line in list(zip(files, kpi, start, end)):
            get_pred_score(line[0],line[1],line[2],line[3], periods=365)

if __name__ == "__main__":
    main()