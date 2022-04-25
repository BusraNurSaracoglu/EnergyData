from enerjisa import a

print("a veri seti yüklendi")


#Test ve train olarak veri setlerimizi ikiye ayırıyoruz.

"""
test_dataset = a[a["Year"] >= 2021 ]
train_dataset = a[a["Year"] < 2021 ]
"""

train_dataset = a[a["Date"] < "01Nov2021 00:00:00"]
test_dataset  = a[a["Date"] >= "01Nov2021 00:00:00"]

print("test veri seti" , test_dataset)
print("train veri seti" , train_dataset)

y_test  = test_dataset["Generation"]
y_train = train_dataset["Generation"]

X_test  = test_dataset.drop(columns=["Generation"])
X_train = train_dataset.drop(columns=["Generation"])

X_test  = X_test.drop(columns=["Date"])
X_train = X_train.drop(columns=["Date"])

print(X_test)