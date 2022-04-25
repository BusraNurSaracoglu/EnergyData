import csv 
#Kütüphaneler
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from pandas import DataFrame

from scipy import stats
import statsmodels.api as sm

from sklearn.preprocessing import MinMaxScaler
from pandas import concat
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.utils import _determine_key_type

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


def row_factory(row):
    return [x if x != '' else 'NaN' for x in row]

"""****************************************************************************************************************************************
@brief       : Verilen verisetlerinin birleştirilmesini sağlayan fonksiyondur.
               Burada 3 adet verisetinin birleştirilmesi gösterilmiştir.
@input   df1 : Veri setinde  en başa gelecek veri setinin adı.
@input   df2 : df1'in yanına eklenecek olan veri seti
@input   df3 : df2'nin yanına eklenecek olan veri setinin adı.
@output  add : Birleştirilen veri seti.
"""

def merge_dfs(df1,df2,df3):
    add = df1.merge(df2,how="left").merge(df3, how="left")
    return add
"""****************************************************************************************************************************************
@brief         : Veri setindeki eksik verilerin sütununun ve sayısının gösterildiği fonksiyondur.
@input   df    : Eksik verileri incelenecek olan veri setinin adı.
@output  eksik : Eksik veriler.
"""

def eksik_veriler(df):
    eksik = df.isna().sum()
    return eksik
"""****************************************************************************************************************************************
@brief                    : Tarih sütunu bilgisine ek olarak diğer bilgilerin ayrışması ve eklenmesi
@input   df               : Tarih bilgilerii ayrılacak olan veri setinin adı.
@output  df['Year']       : Tarih bilgisindeki yıl 
@output  df['Month']      : Tarih bilgisindeki ay
@output  df['Day']        : Tarih bilgisindeki gün
@output  df['WeekOfYear'] : Tarih bilgisindeki yılın hangi haftası bilgisi
"""

def split_date(df):
    df['Date'] = pd.to_datetime(df['DateTime'], format = '%d%b%Y %H:%M:%S')
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['Hour'] = df.Date.dt.hour
    df['dayofweek'] = df.Date.dt.dayofweek
    df['quarter'] = df.Date.dt.quarter
    df['dayofyear'] = df.Date.dt.dayofyear
    df['dayofmonth'] = df.Date.dt.day
    df['weekofyear'] = df.Date.dt.weekofyear

"""****************************************************************************************************************************************
@brief         : Kategorik veriden nümerik veriye çevirme işlemi. 
@input   df    : Veri setinin adı.
@input   col   : İlgili sütun adı.
@output  dummy : Nümerik veriye çevrilmiş hali.
"""


def dummies(df,col):
    dummy = pd.get_dummies(df, columns = [col])
    return dummy
"""****************************************************************************************************************************************
@brief         : Veri setinde bulunan NaN değerlerini sıfıra çeviren fonksiyondur. 
@input   df    : Veri setinin adı.
@input   col   : İlgili sütun adı.
"""

def fillnazero(df,col):
    df[[col]] = df[[col]].fillna(0)
    return
"""****************************************************************************************************************************************
@brief         : Veri setinde bulunan ilgili sütunu düşürme fonksiyonudur. 
@input   df    : Veri setinin adı.
@input   col   : İlgili sütun adı.
@outpt   df1   : Sütunu düşürülmüş dataframe.
"""

def drop_col(df,col):
    df1 = df.drop(columns=[col])
    return df1

"""****************************************************************************************************************************************
@brief            : Veri setinde bulunan tarih kısmını istediğimiz yerden ileri ve geri şekilde bölne fonksiyondur. 
@input   df       : Veri setinin adı.
@input   col      : İlgili sütun adı.
@input   date     : Tarih bilgisi.
@input   select   : İleri zaman ya da geri zaman olarak belirlenen bu kesim işleminde, ileri tarihli bir zaman yazılacaksa 
                    İleri yazılırken, geri zamanlı bir tarih için herhangi bir şey yazılabilir.
@output  ileri    : Verilen tarihin sonrasını getirir.
@output  geri     : Verilen tarihin gerisini getirir.
"""


def split(df,col,date,select):
    geri = df.loc[df[col]<date]
    ileri = df.loc[df[col]>=date]
    if select == "ileri":
        return ileri
    else:
        return geri 
"""****************************************************************************************************************************************
@brief              : Test ve train verisetlerini oluştururken kullanılan veri setini ve kullanılacak olan 
                      satırları çağıran fonksiyondur.
@input   df         : Veri setinin adı.
@input   col_list   : Test veya train için kullanılacak olan sütunların listesi.
@output  df_train   : oluşturulan dataframe.
"""


def create(df,col_list):
    df_train = df[col_list]
    return df_train
"""****************************************************************************************************************************************
@brief            : XGBoost algoritması kullanılarak modelin eğitilmesi ve eğitim sonucu çıkan tahmin değeri olan
                    y_pred değeri için hazırlanmış olan fonksiyondur.
@input   tX       : Train edilecek olan X değerleri.
@input   tY       : Train edilecek olan y değerleri.
@input   testX    : Test edilecek olan  X değerleri
@output  y_pred   : Eğitim sonucu oluşturulan tahmin.
"""

#def XGB(tX,tY,testX):
   # model = XGBRegressor(random_state = 1)
    #model = model.fit(tX,tY)
    #y_pred = model.predict(testX)
   # return y_pred
"""****************************************************************************************************************************************
@brief               : Hata tahminlerinin hesaplanmasını sağlayan fonksiyondur.
@input   testy       : Olması gereken sonuç değerleri.
@input   yhat        : Tahmin edilen sonuç değerleri.
@output  mae_model   : Ortalama Mutlak Hata
@output  mse_model   : Ortalama Kare Hata 
@output  rmse_model  : Kök Ortalama Kare Hata 
"""


def errors(testy,yhat):
    mae_model = mean_absolute_error(testy,yhat)
    mse_model = mean_squared_error(testy,yhat)
    rmse_model = np.sqrt(mse_model)
    
    return print("mae:", mae_model  ,"\nmse:", mse_model, "\nrmse:",rmse_model)
    

"""****************************************************************************************************************************************
@brief              : r2 değerini hesaplayan fonksiyondur.
@input   testyvalue : Olması gereken y değerleri.
@input   yhat       : Tahmin edilen olan y değerleri.
@output  r2_value   : r2 oranı.
"""

def r2(testyvalues, yhat):
    r2_value = r2_score(testyvalues,yhat)
    return r2_value

"""****************************************************************************************************************************************
    @brief            : scatter plot çizdirmek için kullanılan fonksiyondur.
    @input   x        : x ekseni
    @input   y        : y ekseni
    @input   figsizea : çerçeve büyüklüğünün x değeri 
    @input   figsizeb : çerçeve büyüklüğünün y değeri
    @input   xlabel   : x ekseninin adı
    @input   xlabel   : y ekseninin adı
    @input   axis     :
"""


def plot_scatter(x,y,figsizea,figsizeb,xlabel,ylabel,axis):

    plt.figure(figsize=(figsizea,figsizeb))
    plt.scatter(x,y, color="green")
    plt.plot(x,x,color="black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis(axis)

"""**********************************************************************************************************************************************
    @brief            : line plot çizdirmek için kullanılan fonksiyondur.
    @input   figsizex : çerçeve büyüklüğünün x değeri 
    @input   figsizey : çerçeve büyüklüğünün y değeri
    @input   lx       : x ekseni
    @input   ly       : y ekseni
    @input   ci       : gölgelendirme olsun mu (None=olmasın) - default olarak vardır.
    @input   color    : grafiğin rengi
    @input   label    : grafiğin adı

"""


def plot_line(figsizex,figsizey,lx,ly,ci,color,label):
    sns.set()
    y, ax = plt.subplots(figsize=(figsizex,figsizey))
    sns.lineplot(x=lx,y=ly,ci=ci,color=color,label=label)


"""*****************************************************************************************************************************************
    @brief            : 2 line plotu üst üste çizdirmek için kullanılan fonksiyondur.
    @input   figsizea : çerçeve büyüklüğünün x değeri 
    @input   figsizeb : çerçeve büyüklüğünün y değeri
    @input   x1       : 1. grafiğib x ekseni
    @input   y1       : 1. grafiğin y ekseni
    @input   x2       : 2. grafiğin x ekseni
    @input   y2       : 2. grafiğin y ekseni
    @input   ci       : gölgelendirme olsun mu (None=olmasın) - default olarak vardır.
    @input   color1   : 1. grafiğin rengi
    @input   color2   : 2. grafiğin rengi
    @input   label1   : 1. grafiğin adı
    @input   label2   : 2. grafiğin adı
"""


def plotline_con(figsizex,figsizey,x1,y1,x2,y2,ci,color1,color2,label1,label2):
    sns.set()
    y, ax = plt.subplots(figsize=(figsizex,figsizey))
    sns.lineplot(x=x1,y=y1,ci=ci,color=color1,label=label1)
    sns.lineplot(x=x2,y=y2,ci=ci,color=color2,label=label2)

#****************************************************************************************************************************************
"""
    @brief              : Tarih verisini shiftlemek için kullanılan fonksiyondur.
    @input   listshift  : Shiftlenecek sütun bilgilerinin listesi 
    @input   shifted    : Shift edilmiş liste 
    @input   lagnumber  : Shift edilecek gün sayısı
    @output  shifted_df : Shift edilmiş verilerin dataframe hali.

"""

def shift(listshift,shifted,df,lagnumber):
    for i in listshift:
        shifted.append(df[i].shift(lagnumber))
    shifted_df = (pd.DataFrame(shifted)).transpose()
    return shifted_df

#****************************************************************************************************************************************
"""
    @brief           : Birleştirilecek dataframeler için oluşturulmuş bir fonksiyondur.
    @input   df      : 1. dataframe 
    @input   df2     : 2. dataframe 
    @input   join    : 
    @input   axis    : Dataframe'in eklenecek kısmını ifade eder. 0 = aşağı, 1 = soluna.
    @output  df_conc : Birleştirilmiş dataframelerin son hali.

"""

def concat(df,df2,join,axis):
    df_conc = pd.concat([df,df2], join=join, axis=axis)
    return df_conc


#****************************************************************************************************************************************
"""
    @brief                      : Test ve train verilerini normalize etmeye sağlayan fonksiyondur.
    @input   train_data         : train verisi
    @input   test_data          : test verisi
    @output  normtrain_data     : normalize edilmiş train verisi
    @output  normtest_data      : normalize edilmiş test verisi

"""



def normalizedata(train_data,test_data):
    scaler = preprocessing.MinMaxScaler()
    normtrain_data = scaler.fit_transform(train_data)
    normtest_data = scaler.transform(test_data)
    return normtrain_data, normtest_data, scaler 

#****************************************************************************************************************************************



