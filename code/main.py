import os
from run_model import make_pred

def main():
    for k in ['handle_time', 'volume', 'aht']:
        end = ['2020-11-01']*10
        # start = ['2016-03-10','2016-01-20','2017-01-01','2016-04-30','2017-01-20','2017-01-10',
        # '2016-03-10','2016-04-30', '2017-01-01', '2017-01-01']
        start = ['2019-11-01'] *10
        kpi = [k]*10
        files = os.listdir('./data/')
        files = list(filter(lambda x: (x[-3:] == 'csv') and (x!='holidays.csv') , files))

        for line in list(zip(files, kpi, start, end)):
            make_pred(line[0],line[1],line[2],line[3], periods=365)
            # stream = os.popen('python code/run_model.py '+line[0]+' '+line[1]+ ' '+line[2]+' '+line[3]+ ' 200')
            # output = stream.readlines()
            # output

if __name__ == "__main__":
    main()