from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

random_state = 7777

x, y = load_iris(return_X_y = True)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = random_state,
    stratify = y
)

model1 = DecisionTreeClassifier(random_state = random_state)
model2 = RandomForestClassifier(random_state = random_state)
model3 = GradientBoostingClassifier(random_state = random_state)
model4 = XGBClassifier(random_state = random_state)

models = [model1, model2, model3, model4]

for model in models:
    model.fit(x_train, y_train)

    print("-------------------", model.__class__.__name__, "-------------------")
    print('acc :', model.score(x_test, y_test))
    print(model.feature_importances_)

# ------------------- DecisionTreeClassifier(random_state=5555) -------------------
# acc : 1.0
# [0.         0.0125     0.42630737 0.56119263]
# ------------------- RandomForestClassifier(random_state=5555) -------------------
# acc : 1.0
# [0.09522768 0.03243642 0.43290218 0.43943372]
# ------------------- GradientBoostingClassifier(random_state=5555) -------------------
# acc : 1.0
# [0.00071309 0.01353259 0.66926079 0.31649354]
# ------------------- XGBClassifier -------------------
# acc : 0.9666666666666667
# [0.01074596 0.01295084 0.894872   0.0814312 ]