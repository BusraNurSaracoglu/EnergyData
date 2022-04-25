from test_train_datasets import X_test,X_train,y_test,y_train
from functions import errors,r2

from sklearn.ensemble import GradientBoostingRegressor

xgbr = GradientBoostingRegressor( n_estimators=400, learning_rate=0.1, max_depth=5, random_state=0)
xgbr.fit(X_train, y_train)


xgb_acc = xgbr.score(X_test,y_test)*100
print("XGBoost Regressor Accuracy - ",xgb_acc)

y_pred = xgbr.predict(X_test)

errors(y_test,y_pred)
r2(y_test,y_pred)

