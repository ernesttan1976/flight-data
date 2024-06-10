from datetime import datetime
from datetime import timedelta


def download_data(start_date, base_file_location="/content/gdrive/MyDrive/Projects/ads-b/data/"):
    """
    download data for the first day of the month from
    https://samples.adsbexchange.com/readsb-hist/
    Args:
      :param start_date:
      :param base_file_location:
    """
    t = start_date
    end_date = start_date + timedelta(days=1)
    five_seconds = timedelta(seconds=5)

    import requests
    base_url = "https://samples.adsbexchange.com/readsb-hist/"

    while t < end_date:
        print(f"Downloading {base_url}{t:%Y/%m/%d/%H%M%S}Z.json.gz")
        try:
            r = requests.get(f"{base_url}{t:%Y/%m/%d/%H%M%S}Z.json.gz")
        except:
            print(f"Error with {t:%Y/%m/%d/%H%M%S}Z.json.gz")
        t += five_seconds

        with open(f"{base_file_location}{t:%Y%m%dT%H%M%S}Z.json.gz", "wb") as file:
            file.write(r.content)
