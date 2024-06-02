```
pip install pandas numpy matplotlib seaborn plotly geopandas scipy scikit-learn statsmodels requests ipykernel streamlit
```


```
docker run -d -p 8888:8888 -v $(pwd)/notebooks:/home/jovyan/work --name jupyter_container quay.io/jupyter/scipy-notebook:2024-05-27

docker cp /notebooks jupyter_container:/home/jovyan/work



docker run -d -p 8888:8888 --mount type=bind,source=$(pwd)/notebooks,target=/home/jovyan/work --name jupyter_container quay.io/jupyter/scipy-notebook:2024-05-27

```