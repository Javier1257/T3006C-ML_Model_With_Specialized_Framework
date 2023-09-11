from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import tensorflow.keras as tf
import pandas as pd
import numpy as np
import itertools

columns = ['PassengerID', 'Survived', 'Pclass', 'Name', 'Sex',
                 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
df = pd.read_csv('titanic/train.csv', names = columns)
df.head()

print(df.info())

df['Age'] = df['Age'].replace(np.nan, int(np.ceil(df['Age'].mean())))
df['Embarked'] = df['Embarked'].replace(np.nan, 'S')

df = pd.get_dummies(df, columns = ['Sex', 'Embarked'])

df.rename(columns = {'Sex_female' : 'Female', 'Sex_male' : 'Male', 'Embarked_C' : 'Cherbourg', 'Embarked_Q' : 'Queenstown',
                        'Embarked_S' : 'Southampton'}, inplace = True)

df['NbrFamOB'] = df['SibSp'] + df['Parch']

df.drop(['PassengerID', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis = 1, inplace = True)

FINAL_DF = pd.DataFrame(df.copy())
print(FINAL_DF)

X = FINAL_DF[['Pclass', 'Age', 'Fare', 'Female', 'Male', 'Cherbourg', 'Queenstown', 'Southampton', 'NbrFamOB']]
Y = FINAL_DF['Survived']

scaler = StandardScaler()
X  = pd.DataFrame(scaler.fit_transform(X))

split_80p = int(len(df) * .80)
split_10p = int(len(df) * .10)

x_train = X[: split_80p]
x_val = X[split_80p : (split_80p + split_10p)]
x_test = X[(split_80p + split_10p) :]

y_train = Y[: split_80p]
y_val = Y[split_80p : (split_80p + split_10p)]
y_test = Y[(split_80p + split_10p) :]

model = tf.Sequential()

model.add(tf.layers.Dense(units = len(x_train.columns), input_shape = (len(x_train.columns), ), use_bias = True, activation = 'relu'))

BETA = 0.03
model.add(tf.layers.Dense(units = 32, activation = 'relu', kernel_regularizer = l2(BETA)))
model.add(tf.layers.Dense(units = 32, activation='relu', kernel_regularizer = l2(BETA)))
model.add(tf.layers.Dense(units = 32, activation='relu', kernel_regularizer = l2(BETA)))
model.add(tf.layers.Dense(units = 1, activation='sigmoid'))

ALFA = 0.003

optimizer = tf.optimizers.Adam(learning_rate = ALFA)
model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])

print(model.summary())

mini_batch_size = 32
mini_batches = [(x_train[k:k + mini_batch_size], y_train[k:k + mini_batch_size]) for k in range(0, len(x_train), mini_batch_size)]

history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

epochs = 150

for epoch in range(epochs) :
    print("Epoch: ", epoch)
    for mini_batch_X, mini_batch_y in mini_batches:
        loss, accuracy = model.train_on_batch(mini_batch_X, mini_batch_y)

    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose = 1)

    history['loss'].append(loss)
    history['val_loss'].append(val_loss)
    history['accuracy'].append(accuracy)
    history['val_accuracy'].append(val_accuracy)

plt.figure(figsize = (10, 5))
plt.plot(history['loss'], label = 'Train Loss')
plt.plot(history['val_loss'], label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred = (model.predict(x_test) > 0.5).astype("int32")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = ['Not Survived', 'Survived']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f'{cm[i, j]}', horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black', fontsize = 60)

plt.show()

precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Precision: {precision : .2f}')
print(f'Accuracy: {accuracy : .2f}')
print(f'Recall: {recall : .2f}')
print(f'F1: {f1 : .2f}')