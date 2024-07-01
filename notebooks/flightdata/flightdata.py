import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt

import requests
import os

import datetime as dt

from pandas import read_csv
import concurrent.futures

from datetime import datetime

import h3
import geopandas as gpd
import shapely
import plotly.express as px

from tqdm import tqdm
tqdm.pandas()

from pandas import read_pickle
import matplotlib.dates as mdates

import seaborn as sn
import sys

# Latitude and longitude coordinates
home_lat = 1.3472764
home_lng = 103.9104234

# Generate H3 hexagons at a specified resolution (e.g., 9)
resolution = 5

# Indicate the number of rings around the central hexagon
ring_size = 463

# global dataframes to avoid keep loading files
df1=None
df2=None
flights_data=None
hexagon_df=None
flights_df=None
hex_list_df = None

def calculate_hexagon_ids(df):
    """
    Calculate Hexagon IDs for each row(ping) in a DataFrame based on their geographic coordinates.
    Args:
        df (pd.DataFrame): DataFrame containing ADSB data with "lat" and "lon" columns.
        hexagon_df (gpd.GeoDataFrame): GeoDataFrame with hexagon geometries and associated Hexagon IDs.
    Returns:
        pd.DataFrame: The input DataFrame with an additional "Hexagon_ID" column indicating the Hexagon ID for each ping.
    """

    # Create a column Hexagon_ID with the ID of the hexagon
    df['Hexagon_ID'] = None

    df = df.sort_values(['flight','time'])

    # Iterate through the rows in the df DataFrame and calculate counts within each hexagon
    for i, ping in df.iterrows():
        if not isinstance(ping['lat'], float):
            # use location of last ping
            found = False
            j=1
            while not found:
                if isinstance(df.loc[i-j,'lat'],float):
                    df.loc[i,'lat']=df.loc[i-j,'lat']
                    df.loc[i,'lon']=df.loc[i-j,'lon']
                    found=True
                elif isinstance(df.loc[i+j,'lat'],float):
                    df.loc[i,'lat']=df.loc[i+j,'lat']
                    df.loc[i,'lon']=df.loc[i+j,'lon']
                    found=True
                else:
                    j+=1
            continue
        resolution=5   
        result = h3.geo_to_h3(ping["lat"], ping["lon"], resolution)
        # print(f'{ping["lat"]},{ping["lon"]}=>{result}')
        if result != 0:
             df.loc[i, 'Hexagon_ID'] = result
    
    return df

def get_adsb_data(data):
    """
    Fetches data from ADSB web api. Filter adsb_icao. Clean up non-numeric values. 
    Assign Hexagon_IDs to each ping. Convert types per the mapping. Filter where flight is null. 
    Assign "good_bad" is True when NIC >= 7, False when NIC <= 6
    Saves data to csv.
    Args:
        data: Dictionary of url and csv paths
    Returns:
        None

    """

    if (os.path.isfile(data["csv"])):
        print(f"csv exists {data['csv']}")
        return
    else:
        try:
            response = requests.get(data["url"])
            print(f"downloaded {data['url']}")
            json_data = response.json()
            
            df = pd.json_normalize(json_data['aircraft'])

            df = df[df["type"] == 'adsb_icao']
            df['time'] = pd.to_datetime(datetime.fromtimestamp(json_data['now']).strftime("%Y%m%d%H%M%S"))
            df['Hexagon_ID'] = pd.NA
            df['good_bad'] = np.NaN
            df = df[['flight','r','time', 'hex', 'Hexagon_ID', 'alt_baro', 'nic', 'good_bad','lat','lon','type','t','category','version','nac_p','nac_v','track','baro_rate','seen_pos','seen','gs','alt_geom']]	
            df = df[df["type"] == 'adsb_icao']
            df['alt_baro'] = df['alt_baro'].apply(lambda x: 0 if (x == "ground" or x == np.nan ) else float(x))
            df["nic"] = df["nic"].replace(np.NaN,0)

            mapping = {
                "flight": str,
                "r": str,
                "time": "datetime64[ns]",
                "hex": str,
                "Hexagon_ID": str,
                "alt_baro": float,
                "nic": float,
                "good_bad": bool,
                "lat": float,
                "lon": float,
                "type": str,
                "t": str,
                "category": str,
                "version": float,
                "nac_p": float,
                "nac_v": float,
                "track": float,
                "baro_rate": float,
                "seen_pos": float,
                "seen": float,
                "gs": float,
                "alt_geom": float
            }
            df=df.astype(mapping, copy=True)
            df['good_bad'] = df['nic'].apply(lambda x: False if x==np.NaN or x<7 else True)
            df = df[~df["flight"].isnull()]

            df = calculate_hexagon_ids(df)

            if not os.path.exists(os.path.dirname(data["csv"])):
                os.makedirs(os.path.dirname(data["csv"]))
            df.to_csv(data["csv"])

        except Exception as e:
            # Handle other exceptions
            print("Get ADSB Data: An error occurred:", e)


def join_large_file(folder_path,start,end,file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    isHeader = True
    with open(file_path, 'w') as outfile:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            file_datetime = datetime.strptime(file_path, 'csv\\%Y%m%d\\%H%M%SZ.csv')
            print(file_datetime)
            if filename.endswith('.csv') and start <= file_datetime and file_datetime <= end:
                with open(file_path, 'r') as infile:
                    # skip row header
                    if isHeader:
                        outfile.write(infile.read())
                        isHeader = False
                    else:
                        outfile.write(''.join(infile.readlines()[1:]))
                print(f'Joined {file_path}')
        print(f"Saved {file_path}")

def get_flight_data(start_date_time):
    """
    Creates a table of flights and calculates stats for each flight based on aggregates over the rows
    Args:
        start_date_time: start time e.g. dt.datetime(YYYY,MM,DD,HH,MM,SS)
    Global:
        df1: dataframe of raw flight pings from ADSB
        global flights_data
    Returns:
        flights_data: one row for each flight with aggregated fields 'flight', 'bad_count', 'total_count', 'percentage_bad','result'
    """
    global df1
    global flights_data
    flights = df1["flight"].unique()
    flights = flights[1:]
    display(f"Found {flights.shape[0]} flights. Preparing data by flights")
    flights_data = pd.DataFrame(flights, columns=['flight']).progress_apply(lambda x: process_flights_data(x['flight']), axis=1)
    display(flights_data.head(10))
    save_df(flights_data, start_date_time, filetype=FileType.FLIGHTS_DATA, parquet=True)
    return flights_data

def process_flights_data(flight):
    """
    Extract "rows" for this flight from the large df1 table. Calculate counts so that the flight cas be classified as "normal", "equipment failure", "analyse" or "unknown"
    This function is called inside a pandas.apply on flights_data.
    Args:
        flight: string of Flight Identifier
    Globals:
        df1: dataframe of raw flight pings from ADSB
    Returns:
        pd.Series: one row of flights_data dataframe with aggregated fields 'flight', 'bad_count', 'total_count', 'percentage_bad','result'
    """
    global df1
    rows = df1[df1['flight']==flight]
    # print(f"{flight} - {len(rows)} pings")
    bad_count = rows[rows['good_bad'] == False].shape[0]
    total_count = rows.shape[0]
    percentage_bad = bad_count/total_count if total_count!=0 else 0
    result = analyse_data(percentage_bad)
    return pd.Series({'flight': flight, 'bad_count': bad_count, 'total_count': total_count, 'percentage_bad': percentage_bad, 'result': result})

    
def process_data(start_date_time,end_date_time, reload=[False,False,False]):
    """
    Args:
        start_date_time: start time e.g. dt.datetime(YYYY,MM,DD,HH,MM,SS)
        end_date_time: end time e.g. dt.datetime(YYYY,MM,DD,HH,MM,SS)
        reload=[False,False,False]:
            Reload flags are for skipping steps and saving processing time
            reload[0]=True loads data from [date]joined.csv (df1)
            reload[1]=True loads data from [date]flights_data.csv (>45 min) (flights_data)
            reload[2]=True loads data from [date]hexbin.csv (df2)
    Returns:
        [df1, df2, flights_data]
        df1: dataframe of 'flight','r','time', 'hex', 'Hexagon_ID', 'alt_baro', 'nic', 'good_bad',
            'lat','lon','type','t','category','version','nac_p','nac_v','track','baro_rate','seen_pos',
            'seen','gs','alt_geom'
        flights_data: dataframe of  'flight','bad_count', 'total_count', 'percentage_bad', 'result'
        df2: dataframe of 'Hexagon_ID', 'total_count, 'bad_count', 'alt_baro_range', 'time_range', 
            'percentage_bad'

    """
    delta = dt.timedelta(seconds=5)
    start = start_date_time
    global df1
    global df2
    global flights_data
    file_path = os.path.join('csv', 'joined', f'{start_date_time.strftime("%Y%m%d")}joined.csv')

    if not reload[0]:

        data_array = []
        while start < end_date_time:
            data_array.append({
                "url": f'https://samples.adsbexchange.com/readsb-hist/{start.strftime("%Y/%m/%d")}/{start.strftime("%H%M%S")}Z.json.gz',
                "csv": os.path.join('csv',start.strftime("%Y%m%d"), f'{start.strftime("%H%M%S")}Z.csv')
                })
            start += delta

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                list(executor.map(lambda i: get_adsb_data(data_array[i]), range(len(data_array))))
        except Exception as e:
            print("Thread Pool: An error occurred:", e)
            
        join_large_file(os.path.join('csv',start_date_time.strftime("%Y%m%d")),start_date_time,end_date_time,file_path)
        csv_to_parquet(start_date_time, filetype = FileType.JOINED)

    df1 = load_df(start_date_time, filetype=FileType.JOINED, parquet=True)
    
    flights_data = get_flight_data(start_date_time) if not reload[1] else load_df(start_date_time, filetype=FileType.FLIGHTS_DATA, parquet=True)

    df2 = get_hexagon_df(start_date_time) if not reload[2] else load_df(start_date_time, filetype=FileType.HEXBIN, parquet=True)

    return [df1, flights_data, df2]


def get_hexagon_df(start_date_time):
    global df1
    global df2
    df2 = df1.groupby(['Hexagon_ID', 'flight'], as_index=False)[['good_bad','alt_baro','time']].agg(
                            total_count=('good_bad', lambda x: (x == x).sum()),
                            bad_count=('good_bad', lambda x: (x == False).sum()),
                            percentage_bad=('good_bad', lambda x: (x == False).sum()/(x==x).sum()),
                            alt_baro_min=('alt_baro', lambda x: x.min()),
                            alt_baro_max=('alt_baro', lambda x: x.max()),
                            time_min=('time', lambda x: x.min()),
                            time_max=('time', lambda x: x.max())
                            )
    df2 = pd.DataFrame(df2)
    df2["Hexagon_ID"]=df2["Hexagon_ID"].replace("0","NA")
    display(df2.head(10))

    # Apply the function to create new 'lat' and 'lng' columns
    coord = df2['Hexagon_ID'].apply(lambda x: h3.h3_to_geo(x) if x!="NA" else (np.nan, np.nan))
    df2[['lat', 'lon']] = pd.DataFrame(coord.tolist(), index=df2.index)

    # coord = df2['Hexagon_ID'].apply(lambda x: h3.h3_to_geo(x) if x!="NA" else pd.Series([np.nan, np.nan]))
    # df2['lat']=coord[0]
    # df2['lon']=coord[1]
    
    save_df(df2, start_date_time, filetype=FileType.HEXBIN, parquet=True)
    return df2


class FileType:
    JOINED = "joined"
    SAMPLED = "sampled"
    HEXBIN = "hexbin"
    FLIGHTS_DATA = "flights_data"

def save_df(df, start_date_time, filetype=FileType.JOINED, parquet=False):
    directory = 'joined'
    if parquet:
        file_path = os.path.join('csv', directory , f'{start_date_time.strftime("%Y%m%d")}{filetype}.parquet')
        display(f'Saving parquet {file_path}')
        df.to_parquet(file_path) 
    else:
        file_path = os.path.join('csv', directory , f'{start_date_time.strftime("%Y%m%d")}{filetype}.csv')
        display(f'Saving csv {file_path}')
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
            df.to_csv(file_path) 
        else:
            df.to_csv(file_path) 

def load_df(start_date_time, filetype=FileType.JOINED, parquet=False):
    directory = 'joined'
    if parquet:
        file_path = os.path.join('csv', directory , f'{start_date_time.strftime("%Y%m%d")}{filetype}.parquet')
        display(f'Reading parquet {file_path}')
        return pd.read_parquet(file_path)
    else:
        file_path = os.path.join('csv', directory , f'{start_date_time.strftime("%Y%m%d")}{filetype}.csv')
        return pd.concat([chunk for chunk in tqdm(pd.read_csv(file_path, chunksize=1000000), desc=f'Loading csv {file_path}')])

def csv_to_parquet(start_date_time, filetype=FileType.JOINED):
    chunk_size = 1000000
    parquet_writer = pd.DataFrame()
    directory = 'joined'
    if filetype == FileType.HEXBIN:
        directory = 'hexbin'
    csv_path=os.path.join('csv', directory , f'{start_date_time.strftime("%Y%m%d")}{filetype}.csv')
    parquet_path=os.path.join('csv', directory , f'{start_date_time.strftime("%Y%m%d")}{filetype}.parquet')
    print("Converting csv to parquet")
    for i,chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        sys.stdout.write("Conversion progress: %d   \r" % (i) )
        sys.stdout.flush()
        parquet_writer = pd.concat([parquet_writer,chunk])
    parquet_writer.to_parquet(parquet_path)
    display(f"{parquet_path}")

def parquet_to_csv(start_date_time, filetype=FileType.JOINED):
    print("Converting parquet to csv")
    df1=load_df(start_date_time, filetype=filetype, parquet=True)
    save_df(df1, start_date_time, filetype=filetype, parquet=False)


class ResultGroup:
    EQUIPMENT_FAILURE = "equipment_failure"
    NORMAL = "normal"
    ANALYSE = "analyse"
    UNKNOWN = "unknown"

# plot Alt/NIC vs Time
def generate_plots(sample_size, group = ResultGroup.ANALYSE):
    global df1
    global flights_data

    if isinstance(df1, type(None)) or isinstance(flights_data, type(None)):
        print("Generate plots: error no data")
        return

    flights = [str(flight) for flight in flights_data[flights_data['result']==group].sample(sample_size)['flight']]
    for flight in flights:
        flight_df = df1[df1['flight']==flight]
        flight_df.set_index("time")

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(flight_df['time'].astype("datetime64[ns]"), flight_df['alt_baro'], 'bx-', label='alt_baro')
        ax1.tick_params('y', colors='b')
        ax1.set_yticks(np.arange(0, 50000, 5000))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y %H:%M:%S'))

        ax1.set_xlabel('Date-Time', fontsize=10)
        ax1.set_ylabel('Altitude (ft)', color='b', fontsize=10)

        plt.gcf().autofmt_xdate()

        ax2 = ax1.twinx()
        ax2.plot(flight_df['time'].astype("datetime64[ns]"), flight_df['nic'], 'rx-', label='nic')
        ax2.set_ylabel('NIC', color='g', fontsize=10)
        ax2.tick_params('y', colors='g')
        ax2.set_yticks(np.arange(0, 14, 1))

        plt.title(f'Flight:{flight} Altitude and NIC vs Time  (categorised as {group})', fontsize=10)
        fig.legend(loc="lower right")
        plt.show()
        url = f"https://doc8643.com/aircraft/{flight_df.iloc[0]['t']}"
        print(url)
# categorise flights based on NIC
def analyse_data(percentage_bad):
    upper_limit=0.9
    lower_limit=0.1
    if percentage_bad>upper_limit:
        return "equipment_failure"
    elif percentage_bad<lower_limit:
        return "normal"
    elif percentage_bad>=lower_limit and percentage_bad<=upper_limit:
        return "analyse"
    else: 
        return "unknown"

# correlations plot
def generate_correlations(sample_size, group = ResultGroup.ANALYSE):
    global df1
    global flights_data

    if isinstance(df1, type(None)) or isinstance(flights_data, type(None)):
        print("Generate plots: error no data")
        return

    flights = [str(flight) for flight in flights_data[flights_data['result']==group].sample(sample_size)['flight']]
    flights_df=None
    for flight in flights:
        flight_df = df1[df1['flight']==flight]
        flights_df = pd.concat([flights_df,flight_df])    
    
    flights_df.set_index("time")
    selected_columns = ["alt_baro", "nic", "nac_p","nac_v","track","baro_rate","seen_pos","seen","gs","alt_geom"]
    print(flights_df.shape[0])
    display(flights_df.head(10))

    corr_matrix = flights_df[selected_columns].corr(method='pearson')
    ax = plt.axes()
    ax.set_title(f'Pearson cofficient for sample size {sample_size}, \ngroup: {group} \nflights: {flights} ',fontsize=10)
    sn.heatmap(corr_matrix, annot=True, cmap=sn.diverging_palette(220, 20, as_cmap=True))
    plt.show()

# generate hexagons
def get_hexagon_grid(latitude, longitude, resolution, ring_size):
    """
    Generate a hexagonal grid GeoDataFrame centered around a specified location.
    Parameters:
    - latitude (float): Latitude of the center point.
    - longitude (float): Longitude of the center point.
    - resolution (int): H3 resolution for hexagons.
    - ring_size (int): Number of rings to create around the center hexagon.
    Returns:
    - hexagon_df (geopandas.GeoDataFrame): GeoDataFrame containing hexagons and their geometries.
    """

    global hexagon_df

    # Get the H3 hexagons covering the specified location
    center_h3 = h3.geo_to_h3(latitude, longitude, resolution)
    hexagons = list(h3.k_ring(center_h3, ring_size))  # Convert the set to a list

    # Create a GeoDataFrame with hexagons and their corresponding geometries
    hexagon_geometries = [shapely.geometry.Polygon(h3.h3_to_geo_boundary(hexagon, geo_json=True)) for hexagon in hexagons]
    hexagon_df = gpd.GeoDataFrame({'Hexagon_ID': hexagons, 'geometry': hexagon_geometries})
    return hexagon_df

# plot hexagon map    
def create_choropleth_map(alpha=0.5, map_style="carto-positron", data="percentage_bad", limits=[0,0.1,0.5,1], lat=home_lat, lon=home_lng, hex_list=[]):
    """
    Create an interactive choropleth map using Plotly Express.
    Parameters:
    - geojson_df (GeoDataFrame): GeoJSON data containing polygon geometries.
    - data_df (DataFrame): DataFrame containing data to be visualized on the map.
    - alpha (float): Opacity level for the map polygons (0.0 to 1.0).
    - map_style (str): Map style for the Plotly map (e.g., "carto-positron").
    - color_scale (str): Color scale for the choropleth map.
    Returns:
    None
    """
    global df2
    global hexagon_df
    global flights_df
    global hex_list_df
    

    # Hexagon grid around HOME
    if isinstance(hexagon_df, type(None)):
        hexagon_df = get_hexagon_grid(home_lat, home_lng, resolution, ring_size)
    print("hexagons calculated")

    if isinstance(hex_list_df, type(None)):
        hex_list_df = df2[(df2["Hexagon_ID"].isin(hex_list))]
    print("hex_list_df calculated")
    display(hex_list_df.head(10))

    # Merge the GeoJSON data with your DataFrame
    merged_df = hexagon_df.merge(df2 if (len(hex_list)==0) else hex_list_df, on="Hexagon_ID", how="left")
    print(f'merged hexagon_df with {"df2" if (len(hex_list)==0) else "hex_list_df"}')

    merged_df['lat'].astype(float)
    merged_df['lon'].astype(float)
    # merged_df = merged_df.dropna(subset=['lat', 'lon']) if merged_df["lat"].isnull().values.any() else merged_df
    
    display(merged_df.head(10))
    print(f"merged_df: {merged_df.shape[0]}")

    # Create a choropleth map using px.choropleth_mapbox
    fig = px.choropleth_mapbox(
        merged_df,
        geojson=merged_df.geometry,
        locations=merged_df.index,  # Use index as locations to avoid duplicate rows
        color=data,
        color_continuous_scale=[[limits[0], f'rgba(0,255,0,{alpha})'],
                                [limits[1], f'rgba(255,255,0,{alpha})'],
                                [limits[2], f'rgba(255,0,0,{alpha})'],
                                [limits[3], f'rgba(255,0,0,{alpha})']],        
        title="GPS Jam Map",
        mapbox_style=map_style,
        center={"lat": lat, "lon": lon},  # Adjust the center as needed
        zoom=2,
    )

    # Customize the opacity of the hexagons
    fig.update_traces(marker=dict(opacity=alpha))

    # Add hover data for hotel names
    fig.update_traces(customdata=merged_df[["Hexagon_ID","bad_count", "total_count", "percentage_bad", "lat", "lon"]])

    # Define the hover template 
    hover_template = "<b>Hexagon ID:</b> %{customdata[0]}<br><b>Location:</b> %{customdata[4]:.4f},%{customdata[5]:.4f}<br><b>Percentage bad:</b> %{customdata[3]:.3f}<br><b>Total Count:</b> %{customdata[2]}<extra></extra>"
    fig.update_traces(hovertemplate=hover_template)

    # Set margins to 25 on all sides
    fig.update_layout(margin=dict(l=35, r=35, t=45, b=35))
    
    # Adjust the width of the visualization
    fig.update_layout(width=1000) 


    # Create a scatter mapbox plot for the flight data
    flight_fig = px.scatter_mapbox(
        flights_df,
        lat="lat",
        lon="lon",
        hover_name="flight", 
        hover_data=["flight","time","alt_baro", "nic", "nac_p","nac_v","track","baro_rate","seen_pos","seen","gs","alt_geom"], 
        color_discrete_sequence=["blue"],  # Set the color of the flight points
        zoom=2,
    )

    # Overlay the flight data onto the choropleth map
    fig.add_trace(flight_fig.data[0])

    fig.show()

# plot flights on hexagon map, only those flights passing through hexagons in the hex_list
def get_flights_with_hex_list(hex_list):
        
        global flights_df
        global df1

        flights_df=None


        for hexagon_ID in hex_list:
                
                # get the flights in that hexagon
                flights_in_hex = df1[df1['Hexagon_ID']==hexagon_ID]['flight'].unique()

                # extract all the rows for those flights
                for flight in flights_in_hex:
                        flight_df = df1[(df1['flight']==flight) & (df1['Hexagon_ID']==hexagon_ID)]
                        time_start = flight_df["time"].min()
                        time_end = flight_df["time"].max()
                        # flight_df2 contains all pings of the flight between the start and end time that the flight is inside the hex. i.e. it includes pings that have null lat/lon values
                        flight_df2 = df1[(df1['flight']==flight) & (df1['time'].between(time_start, time_end))]
                        flights_df = pd.concat([flights_df,flight_df2])

        display(flights_df.head(5))

        return flights_df
        # plot choropleth map with flights overlaid on top of the hexagons
        create_choropleth_map(data="percentage_bad", limits = [0,0.1,0.5,1], lat=hex_list[0][0], lon=hex_list[0][1], hex_list=hex_list)


# note that for the free sample, only the first day of each month is available on the adsbexchange
# 4 hours + to download 1 day of files
# 18 mins to reload


