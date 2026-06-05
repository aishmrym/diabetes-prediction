from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from train import (
    y_test,
    lr,
    knn,
    dt,
    rf,
    X_test
)

pred_lr = lr.predict(X_test)
pred_knn = knn.predict(X_test)
pred_dt = dt.predict(X_test)
pred_rf = rf.predict(X_test)

for nama, pred in [
    ('Logistic Regression', pred_lr),
    ('KNN', pred_knn),
    ('Decision Tree', pred_dt),
    ('Random Forest', pred_rf)
]:

    print(f"\n{nama}")

    print("Accuracy :", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall   :", recall_score(y_test, pred))
    print("F1 Score :", f1_score(y_test, pred))