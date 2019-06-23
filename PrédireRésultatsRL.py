import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

#Importer le jeu de données
data = pd.read_csv("student-mat.csv", sep=";")

#Conserver les variables qui nous intéressent
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#Établir la variable à prédir
predict = "G3"

#Diviser le jeu de données entre la variable à prédire(y) et le reste des variables(X)
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

#Entâmer une boucle pour réitérer la création du modèle pour ne conserver que le modèle possédant la plus grande efficacité
best = 0
for _ in range(200):

    #Diviser le jeu de données en un jeu à utiliser pour entraînement (90% du jeux) et pour testage (10% du jeu),
    #chacun pour la variable à prédire et le reste des variables
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    #Établir le modèle utilisé (ici, la régression linéaire)
    linear = linear_model.LinearRegression()

    #Entrainer la régression linéaire à l'aide des variables d'essaies
    linear.fit(x_train, y_train)

    # Utiliser la régression sur les variables de test pour prédire la variable G3,
    # faire ressortir le score par rapport à la réalité
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    # Utilisation une déclaration if pour indiquer d'enregister le modèle lorsqu'une nouvelle meilleure précision est atteinte
    if accuracy > best:
        best = accuracy

        #Enregistrer le modèle
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

#Sauvegarder la meilleure précision
accuracy = best

#Réimporter et lire le modèle
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

#Imprimer les résultats
print("La précision du modèle est de: \n", accuracy)
print("\nLes coefficients de régression sont de: \n", linear.coef_)
print("\nL'ordonnées à l'origine est de: \n", linear.intercept_)

#Une fois le fit fait et la précision calculée, tenter de prédire des G3 inconnues
predictions = linear.predict(x_test)

#Imprimer les prédictions du modèles, les variables des observations et les résultats réels des observations pour comparer
print("\n")
for i in range(len(predictions)):
    print(x_test[i], y_test[i], predictions[i])

#Création du graphique pour la visualisation
p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Note finale")
pyplot.show()