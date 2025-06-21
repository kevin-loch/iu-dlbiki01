
# Import der Bibliotheken

import numpy # Bibliothek unter anderem für die Verwendung von Arrays.

# Import von verschiedenen Klassen/Algorithmen aus der Scikit Bibliothek. 
from sklearn.cluster import KMeans # Import K-Means Algorithmus.
from sklearn.svm import SVC # Import Support-Vektor-Machine Algorithmus. 
from sklearn.preprocessing import MinMaxScaler # Import der Klasse MinMaxScaler für Normalisierung der Daten.

from sklearn.metrics import classification_report # Import der Funktionen für die Bewertungsmatrix.
from sklearn.datasets import fetch_openml # Import der Funktionen für das Laden des MNIST Datensatzes.
from sklearn.model_selection import train_test_split # Import der Funktion für Leistungsüberprüfung.
# Pandas zur Darstellung von Daten

# MNIST Datensatz laden
print("MNIST-Datensatz wird geladen...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False) # Angabe des Datensatzes, welches aus https://www.openml.org/search?type=data&status=active&tags.tag=Computer-Vision&id=554 heruntergeladen wird.
X = mnist.data # Enthält Bilddateien (als Matrix).
Y = mnist.target.astype(int) # Umrechnung für spätere Berechnungen, da fetch_openml die einzelnen Bilder als str lädt. Enthält die Werte, welche auf den Bildern zu sehen sind.

# Aufteilung der Daten (60k Training, 10k Testen)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=10000, random_state=42, stratify=Y) # Parameter in richtiger Reihenfolge: Bilddateien(als Matrix), Werte, Anzahl Testdaten, Startwert für Zufallszahlen, stratify für gleichmäßige Verteilung.
# Output
'''
X_train = Bilddateien für Training
X_test = Bilddateien für Testen
Y_train = Werte der Bilddateien für Training (Labels)
Y_test = Werte der Bilddateien für Testen (Labels)
'''
# Normalisieren auf [0 - 1] 
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# K-Means Clustering
print("K-Means wird trainiert...")
kmeans = KMeans(n_clusters=10, init='k-means++', n_init=10, random_state=42)  # Parameter in richtiger Reihenfolge: 10 Cluster aufgrund von 10 Zeichen, Auswahl der Initialisierungsmethode, Anzahl an Durchläufen, Zufallszahlengenerierung.
kmeans.fit(X_train) # Anwendung Modell auf die Trainingsdaten.

# Cluster (Labels werden zugeordnet)
print("Cluster-Labels werden bestimmt...")
labels = {} # Erstellung leeres Dictionary.
for cluster in range(10): # Gehe durch insgesamt 10 Cluster...
    index = numpy.where(kmeans.labels_ == cluster)[0] #Liste mit Indizes.
    if len(index) > 0:
        maj_label = numpy.bincount(Y_train[index]).argmax() # Es werden die Werte der Bilddateien für das Training verwendet. Numpy.bincount zählt, wie oft jede Ziffer darin vorkommt. Argmax gibt den Index mit den meisten Stimmen zurück.
        labels[cluster] = maj_label
    else:
        labels[cluster] = -1  # Falls das Cluster leer sein sollte.

# Testdaten vorhersagen
print("Vorhersage mit k-Means...")
test_clusters = kmeans.predict(X_test) # Mittels der Fuktion wird bestimmt, welche Bilddatei in welches der 10 Cluster fallen würde.

# Es werden die Cluster-IDs in Ziffernlabels umgewandelt. 

pred_kmeans = [] # Leeres Array (Liste) für die Vorhersagen.

# Alle Clusterzuordnungen der Testbilder werden durchlaufen.
for c in test_clusters:
    # Holt das Label, welches diesem Cluster zugeordnet wurde.
    label = labels[c]

    # Fügt es zur Vorhersage-Liste hinzu.
    pred_kmeans.append(label)

# Wandelt die Liste in ein NumPy-Array um, für mathematische Berechnungen.
pred_kmeans = numpy.array(pred_kmeans)

# Support Vector Machine
print("Support Vector Machine wird trainiert...")
svm = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42) 
''' 
Kernel bestimmt Entscheidungsgrenze. Hierbei wird der Standardkernel "rbf" vewendet. Gamma wird verwendet, um zu bestimmten, wie weit der Einfluss eines Trainingspunktes reicht. 
Scale ist Standard. C ist ein Strafparamter. Je kleiner der Wert, desto mehr Fehler werden erlaubt. Hierbei ist 1 ein Standardwert. Anschließend wird der Startwert für Zufallszahlen angegeben.

Training auf kompletten 60k Samples dauert lang.
Subset für schnelleren Test wurde auf 10k Samples gestellt)
'''
samples = 10000
svm.fit(X_train[:samples], Y_train[:samples]) # Modell wird trainiert. Bilddateien und Werte für Bilddateien (Labels) für das Training werden geladen. Samples gibt an, wieviele Zeilen angenommen werden.

print("Vorhersage mit SVM.")
pred_svm = svm.predict(X_test) # Mit der Methode werden neue Vorhersagen getätigt. Hierbei wird die X_test Varialbe verwendet, welche nicht beim Training beachtet wurde. 

# Berichterstellung
print("\nErgebnisse k-Means:")
print(classification_report(Y_test, pred_kmeans, digits=3)) # Paramter Erklärung: Werte für Bilddateien (Labels), Vorhersagen von KMeans, Ausgabe auf 3 Nachkommastellen.

print("\nErgebnisse SVM:")
print(classification_report(Y_test, pred_svm, digits=3)) # Paramter Erklärung: Werte für Bilddateien (Labels), Vorhersagen von SVM, Ausgabe auf 3 Nachkommastellen.

'''
Folgende Dokumentationen wurden für die Erstellung des Quellcodes benötigt:

K-Means: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
SVC: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
SVC Kernel: https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html
MinMaxScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
Classification_report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
Fetch_openml: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html
Train_test_split: https://datascientest.com/de/train-test-split
MNIST Datensatz: https://www.openml.org/search?type=data&sort=runs&id=554&status=active
Matplotlib: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
Matplotlib imshow: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
Tight_Layout: https://matplotlib.org/stable/users/explain/axes/tight_layout_guide.html

'''

