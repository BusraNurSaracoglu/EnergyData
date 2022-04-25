import pandas as pd
from functions import split_date


# Veri setleri import ediliyor.
enerji_df   = pd.read_csv('C:/Data/generation.csv', delimiter=';', decimal=',')
features_df = pd.read_csv('C:/Data/temperature.csv', delimiter=';', decimal=',')

# İndex kısmında bulunan nan değerlerin temizlenmesi
enerji_df = enerji_df.dropna(how="all")
features_df = features_df.dropna(how = "all")


# Veri setlerinin birleştirilmesi
dataset = enerji_df.merge(features_df, how="left")

# Eksik verilerin tespit edilmesi
missing_values = dataset.isnull().sum()
print(missing_values)


# Tarih verilerinin ayrıştırılması
split_date(dataset)


# Datetime index olarak ayarlanıyor.
a = dataset.set_index("DateTime",drop=True)
# a = a.drop(columns="Date")
print(a.index.is_unique)

# Eksik verilerin doldurulması
a["WWCode"] = a["WWCode"].fillna(value = a["WWCode"].mean() )
missing_values = a.isnull().sum()
print(missing_values)

# İndex ismi kaldırılıyor.
a = a.rename_axis(None)


# Veri setinin son hali kontrol ediliyor.
print(a)
