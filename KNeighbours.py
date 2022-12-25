from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=11)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
df_predict = pd.DataFrame({"real": y_test, "predict": y_pred})
df_predict.info()
df_predict.to_csv(r'.\predict.csv', index=False)
print('Процент правильно предсказанных исходов:', round((accuracy_score(y_test, y_pred) * 100), 2), '%')