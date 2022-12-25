from sklearn import tree

from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('train.csv')

z = df['id']

df.drop(
    ["id", "has_photo", "graduation", "education_form", "followers_count", "education_status", "langs", "people_main",
     "city", "last_seen", "occupation_name", "career_start", "career_end"],
    axis=1, inplace=True)


def fill_age(row):
    bdate = str(row["bdate"])
    if pd.isnull(row["bdate"]):
        return row["bdate"]
    if bdate[-2:].isdigit():
        return abs(22 - int(bdate[-2:]))
    return 22 - int(bdate[-1:])


df["bdate"] = df.apply(fill_age, axis=1)


def fill_occup(row):
    occup = str(row["occupation_type"])
    if pd.isnull(row["occupation_type"]):
        if pd.isnull(row["bdate"]):
            return 0
        elif int(row["bdate"]) > 20:
            return 1
        return 0
    if occup == "work":
        return 1
    return 0


df["occupation_type"] = df.apply(fill_occup, axis=1)

age_1 = df[df['occupation_type'] == 1]['bdate'].median()
age_2 = df[df['occupation_type'] == 0]['bdate'].median()


def fill_age2(row):
    if pd.isnull(row["bdate"]):
        if row["occupation_type"] == 0:
            return int(age_2)
        return int(age_1)
    return int(row["bdate"])


df["bdate"] = df.apply(fill_age2, axis=1)

df["sex"] = df["sex"].apply(lambda sex: sex - 1)

df["has_mobile"] = df["has_mobile"].apply(lambda x: int(x))

df["relation"] = df["relation"].apply(lambda x: int(x))


def fill_life(row):
    if row["life_main"].isdigit():
        return int(row["life_main"])
    return 0


df["life_main"] = df.apply(fill_life, axis=1)
df.info()

X = df.drop('result', axis=1)
y = df['result']

model = tree.DecisionTreeClassifier(criterion="gini")
print(tree.DecisionTreeRegressor().get_params())
model.fit(X, y)

print(model.score(X, y))

trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.3)
decision = tree.DecisionTreeClassifier(criterion="gini")
decision.fit(trainX, trainY)
print("Accuracy: "+str(decision.score(testX, testY)))


from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus

dot_data = StringIO()
export_graphviz(model, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('result.png')
