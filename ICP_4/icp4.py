# Import libraries
from sklearn import datasets
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

# Load the information in
iris = datasets.load_iris()

# Split the data up
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

# Train the data and predictions for LinearSVC, setting max iterations higher than the default to avoid
# convergence problems
clf = LinearSVC(max_iter=5000).fit(X_train, y_train)
clf_predict = clf.predict(X_test)

# Train the data and predictions for Naive Bayes
gnb = GaussianNB().fit(X_train, y_train)
gnb_predict = gnb.predict(X_test)

# Print out evaluations for the model predictions
print('\t\t\t\tLinearSVC\t\tNaive Bayes')

# Print out accuracy
print('Accuracy:\t\t{:.2f}'.format(clf.score(X_test, y_test)) + '\t\t\t{:.2f}'.format(gnb.score(X_test, y_test)))

# Print out precision
print('Precision:\t\t{:.2f}'.format(precision_score(y_test, clf_predict, average='macro'))
      + '\t\t\t{:.2f}'.format(precision_score(y_test, gnb_predict, average='macro')))

# Print out recall
print('Recall:\t\t\t{:.2f}'.format(recall_score(y_test, clf_predict, average='macro'))
      + '\t\t\t{:.2f}'.format(recall_score(y_test, gnb_predict, average='macro')))

# Print out f1
print('F1:\t\t\t\t{:.2f}'.format(f1_score(y_test, clf_predict, average='macro'))
      + '\t\t\t{:.2f}'.format(f1_score(y_test, gnb_predict, average='macro')))
