from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

random_state = 7777

x, y = load_diabetes(return_X_y = True)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = random_state
)

model1 = DecisionTreeRegressor(random_state = random_state)
model2 = RandomForestRegressor(random_state = random_state)
model3 = GradientBoostingRegressor(random_state = random_state)
model4 = XGBRegressor(random_state = random_state)

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