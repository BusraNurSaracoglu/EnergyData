from test_train_datasets import X_test,X_train,y_test,y_train
from functions import errors,r2


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100, random_state = 0,)
rf = rf.fit(X_train, y_train)


rf_acc = rf.score(X_test,y_test)*100
print("Random Forest Regressor Accuracy - ",rf_acc)


y_pred = rf.predict(X_test)


errors(y_test,y_pred)
r2(y_test,y_pred)

