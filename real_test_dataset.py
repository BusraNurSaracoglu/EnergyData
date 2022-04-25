from enerjisa import features_df
from functions import split_date


split_date(features_df)


# Datetime index olarak ayarlanıyor.
b = features_df.set_index("DateTime",drop=True)
print(b.index.is_unique)

# Eksik verilerin doldurulması
b["WWCode"] = b["WWCode"].fillna(value = b["WWCode"].mean() )
missing_values = b.isnull().sum()
print(missing_values)

b = b.rename_axis(None)
print(b)


test_dataset  = b[b["Date"] >= "01Dec2021 00:00:00"]
test_dataset  = test_dataset.drop(columns=["Date"])

X_test = test_dataset
print(test_dataset)
