import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
            

def get_columns(file, columns):
    return pd.read_csv(file, sep=';', encoding='utf-8', usecols=columns)

#***************

work_time_columns = [
    "Date",
    "Effectif SNCF"
    ]

work_time = get_columns('./data/effectifs-disponibles-sncf-depuis-1851.csv', work_time_columns)
work_time_filter = work_time[(work_time['Date'].astype(str).between('2015', '2018'))].sort_values(by='Date', ascending=True)

#***************

frequentation_columns = [
    "Nom de la gare",
    "Code postal",
    "Total Voyageurs + Non voyageurs 2020",
    "Total Voyageurs + Non voyageurs 2021",
    "Total Voyageurs + Non voyageurs 2022",
    "Total Voyageurs + Non voyageurs 2023"
    ]

frequentation = get_columns('./data/frequentation-gares.csv', frequentation_columns)
frequentation_filter = frequentation[frequentation['Code postal'].astype(str).str[:2] == '59'].head(10)

# print("Work time: ")
# print(work_time_filter)
print("Frequentation: ")
print(frequentation_filter)

#***************************

def get_frequentation_diagram(frequentation):
    frequentation_df = pd.DataFrame(frequentation)
    frequentation_df.set_index('Nom de la gare')[[
        'Total Voyageurs + Non voyageurs 2015',
        'Total Voyageurs + Non voyageurs 2016', 
        'Total Voyageurs + Non voyageurs 2017', 
        'Total Voyageurs + Non voyageurs 2018']].plot(kind='bar', figsize=(12, 6))
    plt.title('Comparaison fréquentation des gares entre 2015 et 2018')
    plt.ylabel('Total voyageurs')
    plt.xlabel('Gare')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.legend(title='Année')
    plt.show()

#get_frequentation_diagram(frequentation_filter)

#***************************

df = pd.DataFrame(frequentation_filter)
df["Croissance annuelle"] = ((df["Total Voyageurs + Non voyageurs 2023"] - df["Total Voyageurs + Non voyageurs 2022"]) / df["Total Voyageurs + Non voyageurs 2022"]) * 100
df.head()

df["Total Voyageurs + Non voyageurs 2024 (estimé)"] = (
    df["Total Voyageurs + Non voyageurs 2023"] * (1 + df["Croissance annuelle"] / 100)
)

x = df[["Total Voyageurs + Non voyageurs 2023", "Croissance annuelle"]]
y = df["Total Voyageurs + Non voyageurs 2024 (estimé)"]

model = LinearRegression()
model.fit(x, y)

print("Coefficient: ", model.coef_)
print("Intercept: ", model.intercept_)

df["Prédiction 2024"] = model.predict(x)
print(df[["Nom de la gare", "Prédiction 2024"]])
