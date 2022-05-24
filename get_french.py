import numpy as np
import pandas as pd
import os
import gzip
import requests


url_stations = "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/postesSynop.csv"
if os.path.exists("temp_file.csv") is False:
    with open("temp_file.csv", "wb") as fid:
        file_stations = requests.get(url_stations)
        fid.write(file_stations.content)

fpd_st = pd.read_csv("temp_file.csv", delimiter=";")
fpd_st = fpd_st[0:40]  # Mainland only
os.remove("temp_file.csv")


year, month = "2020", "01"
url_temp = "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/Archive/synop.{}{}.csv.gz".format(year, month)
if os.path.exists("temp_file.csv") is False:
    zipped_content = requests.get(url_temp).content
    with open("temp_file.csv", "wb") as fid:
            fid.write(gzip.decompress(zipped_content))

fpd_tp = pd.read_csv("temp_file.csv", delimiter=";")
os.remove("temp_file.csv")

# Preprocessing: drop useless data, cast to numeric datatypes
fpd_tp = fpd_tp[["numer_sta", "date", "t"]]  # keep useful columns
fpd_tp = fpd_tp[fpd_tp["numer_sta"].isin(fpd_st.ID)]  # keep metropolitan stations
fpd_tp = fpd_tp.replace({"mq": "nan"})  # replace mq by nan
fpd_tp = fpd_tp.astype({"t": float})  # convert temperatures to float

year, month, day, hour = year, month, "01", "03" # 2020-01-01 @03:00 am

date = int("".join((year, month, day, hour, "00", "00"))) #AAAAMMDDHHMMSS  
temp = fpd_tp[fpd_tp.date == date]

temps_data=fpd_st.merge(temp,left_on='ID',right_on='numer_sta')
temps_data.to_csv('french_2d.csv', sep=';')

name_tls = "TOULOUSE-BLAGNAC"
id_tls = int(fpd_st.ID[fpd_st.Nom == name_tls])
fpd_tls = fpd_tp[fpd_tp.numer_sta == id_tls]  # data toulouse
time_formatted = pd.to_datetime(fpd_tls.date, format="%Y%m%d%H%M%S")
fpd_tls.to_csv('french_1d.csv', sep=';')

