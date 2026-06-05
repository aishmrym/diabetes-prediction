from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocessing import X_scaled, y

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Logistic Regression
lr = LogisticRegression(
    solver='lbfgs',
    max_iter=2000,
    C=0.5,
    class_weight='balanced'
)

lr.fit(X_train, y_train)

# KNN
knn = KNeighborsClassifier(
    n_neighbors=3,
    weights='distance',
    metric='minkowski',
    p=1
)

knn.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    min_samples_split=4,
    class_weight='balanced'
)

dt.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_split=4,
    max_features='sqrt',
    class_weight='balanced_subsample',
    random_state=42
)

rf.fit(X_train, y_train)

print("Training selesai")