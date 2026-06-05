import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from evaluate import (
    pred_lr,
    pred_knn,
    pred_dt,
    pred_rf,
    y_test
)

hasil = pd.DataFrame({
    'Model':[
        'Logistic Regression',
        'KNN',
        'Decision Tree',
        'Random Forest'
    ],
    'Accuracy':[
        accuracy_score(y_test,pred_lr),
        accuracy_score(y_test,pred_knn),
        accuracy_score(y_test,pred_dt),
        accuracy_score(y_test,pred_rf)
    ]
})

sns.barplot(
    x='Model',
    y='Accuracy',
    data=hasil
)

plt.show()