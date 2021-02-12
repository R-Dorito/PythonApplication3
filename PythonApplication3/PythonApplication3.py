
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

#Tells us how many
print(dataset.shape)
#Shows the first 20 data points
print(dataset.head(20))
#Returns the mean standard dev and others 
print(dataset.describe())
#What classes does each data belong to, and what size
print(dataset.groupby('class').size())

#Create a box plot for each item
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#Histogram
dataset.hist()
#Matrix scatter plot. Not too sure what this one is
scatter_matrix(dataset)
pyplot.show()

