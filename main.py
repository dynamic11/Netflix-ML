import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

# enviroment booleans
DEBUG = True
PLOTS = True

# read data form file CSV into pandas data structure
training_data = pd.read_csv('data/CSV/netflix-train.csv', sep=';')
testing_data = pd.read_csv('data/CSV/netflix-test.csv', sep=';')

training_y = training_data['expected']
training_x = training_data.drop('expected', axis=1)

testing_y = testing_data['expected']
testing_x = testing_data.drop('expected', axis=1)

# generate MLP with 1 hidden layer of 9 neurons (18input/2)
mlp = MLPClassifier(hidden_layer_sizes=(15,),
                    max_iter=500,
                    activation = 'relu',
                    solver='adam',
                    verbose=DEBUG,
                    early_stopping=True,
                    validation_fraction=0.2,
                    )
mlp.fit(training_x, training_y)

# plot the error curve
plt.plot(mlp.loss_curve_,  label='Model Error vs Epoch')
plt.title('Learning Loss Function')
plt.xlabel('Loss')
plt.ylabel('Epoch')
plt.savefig("output/accuracy_vs_epoch.png", bbox_inches='tight', dpi=200, pad_inches=0.5)
plt.close()

# run the test data to see validity of model
predictions = mlp.predict(testing_x)
cnf_matrix = confusion_matrix(testing_y, predictions)

print("***************************************")
print(cnf_matrix)
print("***************************************")


# plot the Confusion matrix
fig, ax = plt.subplots(1)
ax = sns.heatmap(cnf_matrix, ax=ax, cmap=plt.cm.Greens, annot=True)
plt.title('Confusion matrix of random forest predictions')
plt.ylabel('True category')
plt.xlabel('Predicted category')
plt.savefig("output/Confusion_Matrix.png", bbox_inches='tight', dpi=200, pad_inches=0.5)
plt.close()

print("Training set score: %f" % mlp.score(training_x, training_y))
print("Test set score: %f" % mlp.score(testing_x, testing_y))

print(mean_squared_error(testing_y, predictions))