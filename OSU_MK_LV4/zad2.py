import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# a
plt.figure()
customCmap = matplotlib.colors.ListedColormap(["red", "blue"])
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=customCmap)
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap=customCmap,marker='x')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("Podaci za treniranje(.) i testiranje(x)")
plt.legend(["x*2" , "x*3"])


# b
LogRegression_model = LogisticRegression ()
LogRegression_model.fit( X_train , y_train )

# c
# u .coef_ se nalaze fi1 i fi2 dok je fi0 zapisan u .intercept_ jer oznacava presjek s y-osi
print(LogRegression_model.coef_)
th = LogRegression_model.coef_[0]

# granica odluke definirana kao x2 = -x1*fi1/fi2 - fi0/fi2
a = -th[0] / th[1]
x1 = np.linspace(-5, 5)

x2 = a * x1 - (LogRegression_model.intercept_[0]) / th[1]

plt.plot(x1, x2, 'k-')
plt.show()

#d
y_pred = LogRegression_model . predict ( X_test )

cm = confusion_matrix( y_test , y_pred )
print (" Matrica zabune:\n" , cm )
disp = ConfusionMatrixDisplay ( confusion_matrix ( y_test , y_pred ) )
disp.plot ()
plt.show ()

print("Accuracy: " , accuracy_score(y_test , y_pred ))
print("Precision: ", precision_score(y_test , y_pred ))
print("Recall: ", recall_score(y_test , y_pred))


# e

customCmap = matplotlib.colors.ListedColormap(["red", "blue"])
customCmap2 = matplotlib.colors.ListedColormap(["orange", "green"]) #Black and Green are too similar
y_isTrue = np.arange(len(y_test))

for i in range(len(y_test)):
    if(y_test[i] == y_pred[i]):
        y_isTrue[i] = 1
    else:
        y_isTrue[i] = 0

plt.figure()
plt.scatter(X_test[:,0],X_test[:,1],c=y_isTrue,cmap=customCmap)
plt.show()
