from functions import download_data
from dateutil.relativedelta import relativedelta
from datetime import datetime
import os


if __name__ == '__main__':
    start_date = datetime(2024, 4, 1)
    end_date = datetime(2024, 5, 1)
    download_data(start_date, os.path.join(os.getcwd(), "data\\"))