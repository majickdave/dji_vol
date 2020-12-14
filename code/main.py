import os
from run_model import *
from split_dataset import *

def main():
    """
    TODO
    add tuple to date_dict for different kpis

    Execute model scoring and prediction pipeline
    """
    # load the most recent data from excel
    df = load_rename('./data/full_dataset.csv')
    # create csv for individual business unit
    write_data(df)
    # when to start training for each business unit
    date_dict = {'BRO-Complex_aht_vol.csv':'2018-10-01', 'BRO_aht_vol.csv': '2018-11-01', 
    'CS-Alaska_aht_vol.csv':'2016-01-01', 'CS-John-Hancock_aht_vol.csv':'2019-12-01', 
    'CS-Maryland_aht_vol.csv': '2017-11-01', 'CS-National_aht_vol.csv':'2016-11-01', 
    'PSG_aht_vol.csv':'2019-11-01', 'RCS-PHONES_aht_vol.csv':'2018-11-01', 
    'RIS_aht_vol.csv':'2019-11-01', 'WISE-Contractual_aht_vol.csv':'2016-11-01'}
    def run_prediction(num_bus=10, 
    kpis=['handle_time', 'volume', 'aht']):
        """
        obtain predictions and score logging for 10 business units
        """
        for j, k in enumerate(kpis):
            # set end date for training data for
            end = ['2020-12-01']*num_bus
            # set start date for training data
            # start = ['2016-01-01'] *num_bus
            kpi = [k]*num_bus
            files = os.listdir('./data/')
            files = list(filter(lambda x: (x[-7:] == 'vol.csv'), files))

            for i, line in enumerate(zip(files, kpi, end)):
                print('*'*50)
                print(k)
                print(line[0])
                print(int(str(j)+str(i))+1,'/',len(files)*len(kpis))
                print('*'*50)
                start = date_dict[line[0]]
                get_pred_score(line[0],line[1],start,line[2], periods=365)

    run_prediction()

if __name__ == "__main__":
    main()