from enerjisa import a
from enerjisa import features_df


train_dataset = a


print("test veri seti" , test_dataset)
print("train veri seti" , train_dataset)

y_test  = test_dataset["Generation"]
y_train = train_dataset["Generation"]

X_test  = test_dataset.drop(columns=["Generation"])
X_train = train_dataset.drop(columns=["Generation"])

X_test  = X_test.drop(columns=["Date"])
X_train = X_train.drop(columns=["Date"])
