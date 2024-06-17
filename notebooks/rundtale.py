import dtale
import dtale.app
import dtale.global_state as global_state
global_state.set_app_settings(dict(enable_web_uploads=True))

import pandas as pd
pd.set_option('display.max_rows', None)

import numpy as np


import os

import datetime as dt
from datetime import datetime

# import time
from pandas import read_csv

start_date_time = dt.datetime(2024,6,1,0,0,0)
csv_path = os.path.join(os.getcwd(), 'csv', 'joined', f'{start_date_time.strftime("%Y%m%d")}joined.csv')
count = 0
with open(csv_path) as fp:
      for _ in fp:
            count += 1
      
nrows = 80000000
df1 = read_csv(csv_path, verbose=True, skiprows=0, nrows=nrows)
print(f"loaded {nrows} of {count} rows")

if __name__ == '__main__':
      dtale.show(df1,subprocess=False, host="127.0.0.1", port="40000", enable_web_uploads=True)
# hexbin_path = os.path.join(os.getcwd(),'notebooks','csv', 'hexbin', f'{start_date_time.strftime("%Y%m%d")}hexbin.csv')
# df2 = read_csv(hexbin_path, verbose=True)

# cli command
# dtale --csv-path /home/jdoe/my_csv.csv --csv-parse_dates date
# dtale --host localhost --csv-path C:\Users\ernes\Raid\flight-data\notebooks\csv\joined\20240601joined-cleaned.csv --csv-parse_dates time