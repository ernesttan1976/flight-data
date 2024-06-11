** Reference
API reference for ADS-B Exchange
https://www.adsbexchange.com/version-2-api-wip/

Mode-S.org by Junzi Sun
https://mode-s.org/decode/content/ads-b/7-uncertainty.html
https://mode-s.org/decode/book-the_1090mhz_riddle-junzi_sun.pdf

** Commands

*** 
```
python -m venv venv
venv/scripts/activate
pip install pandas numpy matplotlib seaborn plotly geopandas scipy scikit-learn statsmodels requests ipykernel streamlit
```

*** Run Jupyter Notebooks as a docker service
```
docker run -d -p 8888:8888 --mount type=bind,source=$(pwd)/notebooks,target=/home/jovyan/work --name jupyter_container quay.io/jupyter/scipy-notebook:2024-05-27
```

Apache Spark with Jupyter Notebooks
```
docker run -d -p 8888:8888 --mount type=bind,source=$(pwd)/notebooks,target=/home/jovyan/work --name jupyter_container jupyter/pyspark-notebook
```



Save your notebooks in ./notebooks
If you want, you can serve the notebooks at port 8888



python -m virtualenv -p C:\Users\ernes\AppData\Local\Programs\Python\Python311 venv