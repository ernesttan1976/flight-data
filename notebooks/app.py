import streamlit as st
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import numpy as np

import matplotlib.pyplot as plt

import requests
import os

import datetime as dt
import time

from pandas import read_csv
import concurrent.futures
import psutil

from datetime import datetime

import h3
import geopandas as gpd
import shapely
import plotly.express as px

if 'filename' not in st.session_state:
    st.session_state['filename'] = None

# Set page config to full width and height
st.set_page_config(
    page_title="SwiftScrub",
    page_icon="🫧",
    layout="wide",
)

# Custom CSS to style the scrollable containers
scrollable_css='''
<style>
    section.main>div {
        padding-bottom: 0rem;
    }
    # [data-testid="stVerticalBlock"]>[data-testid="stHorizontalBlock"]:has([data-testid="stMarkdown"]){
    #     overflow: auto;
    #     max-height: 650px;
    # }
    # [data-testid="element-container"] [data-testid="stTable"]{
    #     overflow: auto; 
    #     max-height: 350px;
    # }
    [data-testid="stExpanderDetails"]:has([data-testid="stTable"]){
        overflow: auto; 
        max-height: 350px;
    }
</style>
'''

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

    # Get the H3 hexagons covering the specified location
    center_h3 = h3.geo_to_h3(latitude, longitude, resolution)
    hexagons = list(h3.k_ring(center_h3, ring_size))  # Convert the set to a list

    # Create a GeoDataFrame with hexagons and their corresponding geometries
    hexagon_geometries = [shapely.geometry.Polygon(h3.h3_to_geo_boundary(hexagon, geo_json=True)) for hexagon in hexagons]
    hexagon_df = gpd.GeoDataFrame({'Hexagon_ID': hexagons, 'geometry': hexagon_geometries})

    return hexagon_df

# Latitude and longitude coordinates
home_lat = 1.3472764
home_lng = 103.9104234

# Generate H3 hexagons at a specified resolution (e.g., 9)
resolution = 5

# Indicate the number of rings around the central hexagon
ring_size = 463



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

    # Iterate through the hotels in the df DataFrame and calculate hotel counts within each hexagon
    for i, ping in df.iterrows():
        if not isinstance(ping['lat'], float):
            continue  
        resolution=5   
        result = h3.geo_to_h3(ping["lat"], ping["lon"], resolution)
        # print(f'{ping["lat"]},{ping["lon"]}=>{result}')
        if result != 0:
             df.loc[i, 'Hexagon_ID'] = result
    
    return df

def create_choropleth_map(geojson_df, data_df, alpha=0.3, map_style="carto-positron", data="percentage_bad", limits=[0,0.1,0.5,1]):
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
    # Merge the GeoJSON data with your DataFrame
    merged_df = geojson_df.merge(data_df, on="Hexagon_ID", how="left")

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
        center={"lat": home_lat, "lon": home_lng},  # Adjust the center as needed
        zoom=2,
    )

    # Customize the opacity of the hexagons
    fig.update_traces(marker=dict(opacity=alpha))

    # Add hover data for hotel names
    fig.update_traces(customdata=merged_df[["Hexagon_ID","bad_count", "total_count", "percentage_bad", "lat", "lng"]])

    # Define the hover template 
    hover_template = "<b>Hexagon ID:</b> %{customdata[0]}<br><b>Location:</b> %{customdata[4]:.4f},%{customdata[5]:.4f}<br><b>Percentage bad:</b> %{customdata[3]:.3f}<br><b>Total Count:</b> %{customdata[2]}<extra></extra>"
    fig.update_traces(hovertemplate=hover_template)

    # Set margins to 25 on all sides
    fig.update_layout(margin=dict(l=35, r=35, t=45, b=35))
    
    # Adjust the width of the visualization
    fig.update_layout(width=1000) 

    fig.show()

def time_elapsed(start_time):
    end_time = time.time()
    return end_time - start_time

time_data={
    "pickle": 0,
    "url": 0,
    "concat": 0,
}

def get_adsb_data(data):
    response = requests.get(data["url"])
    with st.echo():
        print(f"downloaded {data['url']}")
    json_data = response.json()
    
    df = pd.json_normalize(json_data['aircraft'])
    df = df.fillna("Nan values")
    df = df[df["type"] == 'adsb_icao']
    df['time'] = datetime.fromtimestamp(json_data['now']).strftime("%Y%m%d%H%M%S")
    df['good_bad'] = df['nic'].apply(lambda x: 'bad' if x=="Nan values" or int(x)<=6 else 'good')
    df['alt_baro'] = df['alt_baro'].apply(lambda x: 0 if (x == "ground" or x == "Nan values" ) else float(x))
    df["nic"] = df['nic'].apply(lambda x: int(float(x)) if x !="Nan values" else 0)


    df['Hexagon_ID'] = None
    df = df[['Hexagon_ID', 'r', 'time', 'alt_baro', 'nic', 'good_bad', 'hex','lat','lon','type','flight','t','category','version','nac_p','nac_v']]
    df = df.map(lambda x: ', '.join(x) if isinstance(x, list) else x)
    df = df[df["r"]!="Nan values"]

    df = calculate_hexagon_ids(df)

    #save to pickle
    if not os.path.exists(os.path.dirname(data["csv"])):
        os.makedirs(os.path.dirname(data["csv"]))
    df.to_csv(data["csv"])
    return df

def download_data(index, data, length_data_array, start_time_overall):
    start_time = time.time()

    try:
        # if pickle exists, do nothing
        if (os.path.isfile(data["csv"])):
            print(f"csv exists {data['csv']}")
            return
        else:
            get_adsb_data(data)

            time_data["url"] += time_elapsed(start_time)

            remaining_time = (time.time() - start_time_overall)/(index+1) * (length_data_array - index)
            with st.echo():
                print(f"Remaining execution time: {remaining_time//60:.0f}m {remaining_time%60:.0f}s")

            return

    except Exception as e:
        # Handle other exceptions
        with st.echo():
            print("An error occurred:", e)

def join_large_csv(folder_path,start,end,file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    isHeader = True
    with open(file_path, 'w') as outfile:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            file_datetime = datetime.strptime(file_path, 'csv\\%Y%m%d\\%H%M%SZ.csv')
            with st.echo():
                print(file_datetime)
            # csv_path = os.path.join('csv',start.strftime("%Y%m%d"), f'{start.strftime("%H%M%S")}Z.csv')
            if filename.endswith('.csv') and start <= file_datetime and file_datetime <= end:
                with open(file_path, 'r') as infile:
                    # skip row header
                    if isHeader:
                        outfile.write(infile.read())
                        isHeader = False
                    else:
                        outfile.write(''.join(infile.readlines()[1:]))
                with st.echo():
                    print(f'Joined {file_path}')

def sampling(datestring, file_path):
    df1 = read_csv(file_path)
    df1_bad = df1[df1['good_bad']=='bad']
    df1_bad.to_csv(os.path.join('csv','joined', f'{datestring}bad.csv'))

    df1_good = df1[df1['good_bad']=='good']
    df1_good.to_csv(os.path.join('csv','joined', f'{datestring}good.csv'))

    sampled_file_path = os.path.join('csv','joined', f'{datestring}sampled.csv')
    bad_path = os.path.join('csv','joined', f'{datestring}bad.csv')
    good_path = os.path.join('csv','joined', f'{datestring}good.csv')
    with open(sampled_file_path, 'w') as outfile:
        with open(bad_path, 'r') as infile:
            outfile.write(infile.read())
        with open(good_path, 'r') as infile:
            lines = infile.readlines()
            filtered_lines = [line for i, line in enumerate(lines[1:]) if (i + 1) % 10 != 0]
            outfile.write(''.join(filtered_lines))
        with st.echo():
            print(f'Joined {file_path}')
    df1 = read_csv(sampled_file_path)

    df1 = df1.sort_values(by=["Hexagon_ID","r","time"], ascending=[True,True,True])
    df1.to_csv(sampled_file_path)

def load_data():
    start_date_time = st.session_state["start_date"] 
    reload=[False, False]
    delta = dt.timedelta(seconds=5)
    start = start_date_time
    end_date_time = start_date_time + dt.timedelta(days=1)
    df1 = None
    df2 = None
    csv_path = os.path.join('csv', 'joined', f'{start_date_time.strftime("%Y%m%d")}joined.csv')
    hexbin_path = os.path.join('csv', 'hexbin', f'{start_date_time.strftime("%Y%m%d")}hexbin.csv')
    sampled_path = os.path.join('csv', 'joined', f'{start_date_time.strftime("%Y%m%d")}sampled.csv')

    if not reload[0]:

        data_array = []
        while start < end_date_time:
            data_array.append({
                "url": f'https://samples.adsbexchange.com/readsb-hist/{start.strftime("%Y/%m/%d")}/{start.strftime("%H%M%S")}Z.json.gz',
                "pickle": os.path.join('pickle',start.strftime("%Y%m%d"), f'{start.strftime("%H%M%S")}Z.pkl'),
                "csv": os.path.join('csv',start.strftime("%Y%m%d"), f'{start.strftime("%H%M%S")}Z.csv')
                })
            start += delta

        report1=""
        report2=""

        start_time_overall = time.time()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            l = [len(data_array)] * len(data_array)
            list(executor.map(lambda i: download_data(i, data_array[i], l[i], start_time_overall), range(len(data_array))))
            report1 = get_memory_usage()

        join_large_csv(os.path.join('csv',start_date_time.strftime("%Y%m%d")),start_date_time,end_date_time,csv_path)
        sampling(start_date_time.strftime("%Y%m%d"), csv_path)

        start_time_overall = time.time()
        with st.echo():
            print(f"Download memory usage\n{report1}\n\n")
            print(f"CSV memory usage\n{report2}")
            # only when all the threads are complete, then concat happens
            print(f'Total directory size: {get_directory_size("csv"):.1f} MB')
            print(f"Download time: {(time_data['url']/3600):.1f} min")
            print(f"CSV time: {(time_data['pickle']/3600):.1f} min")
    
    with st.echo():
        print("loading csv")
    df1 = read_csv(sampled_path)
    with st.echo():
        print("csv loaded")

    if not reload[1]:

        df2 = groupby_hexagon_ID(df1, hexbin_path)
    
    else:
        df2 = read_csv(hexbin_path)

    return [df1, df2]

def groupby_hexagon_ID(df1, hexbin_path):
    df2 = df1.groupby(['Hexagon_ID', 'r'], as_index=False).agg(
                            good_count=('good_bad', lambda x: (x == 'good').sum()),
                            bad_count=('good_bad', lambda x: (x == 'bad').sum()),
                            alt_baro_range=('alt_baro', lambda x: (x.max() - x.min())),
                            time_range=('time', lambda x: x.max() - x.min()))

    df2["total_count"]=df2["good_count"]+df2["bad_count"]
    df2["percentage_bad"]=df2["bad_count"]/df2["total_count"]
    def get_lat(row):
        return h3.h3_to_geo(row['Hexagon_ID'])[0]

    def get_lng(row):
        return h3.h3_to_geo(row['Hexagon_ID'])[1]

    # Apply the function to create new 'lat' and 'lng' columns
    df2['lat'] = df2.apply(get_lat, axis=1)
    df2['lng'] = df2.apply(get_lng, axis=1)

    unique = df1.groupby('Hexagon_ID', as_index=False)['r'].nunique().sort_values('r',ascending=False)
    unique = unique.rename(columns={"r": "r_count"})

    df2 = df2.merge(unique, on='Hexagon_ID', how='left').sort_values(["r_count", "Hexagon_ID"], ascending=[False,True])
    df2 = df2[['Hexagon_ID', 'r_count',	'r', 'good_count','bad_count','alt_baro_range','time_range','total_count','percentage_bad']]
    df2 = df2.groupby('Hexagon_ID')[['r_count','r', 'good_count','bad_count','alt_baro_range','time_range','total_count','percentage_bad']]
    df2.sum().reset_index().to_csv(hexbin_path)

    return df2

def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
            total_size_mb = total_size / (1024 * 1024)
    return total_size_mb

def get_memory_usage():

    # Get the memory usage
    memory_usage = psutil.virtual_memory()

    # Print the memory usage
    report = f"""
        Total Memory: {memory_usage.total / (1024 ** 3):.2f} GB
        Available Memory: {memory_usage.available / (1024 ** 3):.2f} GB
        Used Memory: {memory_usage.used / (1024 ** 3):.2f} GB
        Memory Usage Percentage: {memory_usage.percent:.2f}%
    """
    return report

# note that for the free sample, only the first day of each month is available on the adsbexchange
# 4 hours + to download 1 day of files
# 18 mins to reload

if 'filename' not in st.session_state:
    st.session_state['filename'] = None

c1= st.container()
c1.markdown("### GPS Jam Exploratory Data Analysis")
start_date = c1.date_input("Select date", datetime.date(2024, 6, 1))
c1.button("Load Data", on_click=load_data, use_container_width=True)

if start_date:
    st.session_state["start_date"]=start_date

@st.experimental_memo
def get_df1():

    return df

def handle_plot_map():
    # Hexagon grid around HOME
    hexagon_df = get_hexagon_grid(home_lat, home_lng, resolution, ring_size)

    with st.echo():
        print("hexagons calculated")
    create_choropleth_map(geojson_df=hexagon_df, data_df=df2, data="percentage_bad", limits = [0,0.1,0.5,1])

show_df1 = c1.dataframe()
show_df2 = c1.dataframe()

# select date
# load df1(...sampled.csv) and df2(...hexbin.csv) 

# plot df2 on choropleth world map

# when user selects a hexagon, show the flights for that hexagon as filter toggle buttons
# user toggles one or all flights
# filter df1 to show the plots for the selected flights in that hexagon
# plots: 
# (1) altitude and nic vs time 
# (2) altitude vs nic

# The idea is to determine if the points are valid, rule out cases of equipment failure (totally no signal)
# if points are valid, then analyse further
# does only one aircraft show bad signals or all aircraft equally show bad signals?
# Is there a continuous time period where all the signals are bad? how long is that period?
# If bad signals are intermittent (not continuous) and not affect all aircraft at all times, 
# then it is safe to say it is not a GPS jammed area