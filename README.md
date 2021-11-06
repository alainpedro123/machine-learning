Machine Learning in action 💻🚀
===============================

Building a simple model (engine) that can learn and predict the kind of music people like. Having an online music store, whenever users sign up we ask for the age and gender, based on their profile the model recommends various types of music they’re likely to buy.

Using Machine Learning in project we're building and feed the model with some sample data based on the existing users, the model will learn the pattern of a data, so we can ask it to make predictions.

When a new user signs up we let it up to the model to decide the type of music this new user is interested in and based on we can make suggestions to the user.


Step 1
----
**Import the data**
```python
import pandas as pd
music_data = pd.read_csv('music.csv')
```

Step 2
----
**Clean the Data and split the Data into two**

* Remove the duplicate data, non-value to prevent our model from learning bad pattern and produce the wrong results
* Preparing the data

**Split the Data into: Training (80%) & Testing (20%)**

Splitting the data into two separate datasets:
* input set (x) – the first 2 columns (age, agender)

```python
# create the input set - create a new dataset without "genre" columns

x = music_data.drop(columns=['genre']) 
```

* output set (y) – the last column (genre)

```python
# create the output set - create a new dataset only with "genre" column

y = music_data('genre') 
```

Step 3
----
**Creating a Model, Training the model, Making Prediction**

```python
# Importing the chosen algorithm – the decision tree
from sklearn.tree import DecisionTreeClassifier

# create a new model
model = DecisionTreeClassifier()

# Training the Model
model.fit(X, y)

# making a prediction
predictions = model.predict([ [21, 1], [22, 0] ])
predictions
```

Step 4
----
**Evaluate and improve: calculating and measuring the accuracy of a model**

```python
# Importing the “training and testing” library
from sklearn.model_selection import train_test_split

# Allocating 80% for training and 20% the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Importing the “Accuracy” library
from sklearn.metrics import accuracy_score

# calcuting the accuracy from 0 to 1
score = accuracy_score(y_test, predictions)
```

Step 5
----
**Putting all together**

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
```

Step 6
----
**Model Persistence**

For large amount of Dataset, we build and train a model and save it to a file

```python
# Creating the the model, training and saving it into "jobLib" file
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib as jb

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X, y)

jb.dump(model, 'music-recommender.joblib')

# Loading the model and checking for Model persistance
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib as jb

model = jb.load('music-recommender.joblib')
predictions = model.predict([[21, 1]])
predictions
```

Libraries and Tool for Machine Learning
----
* NumPy 
* Pandas 
* MatPlotLib
* Scikit-Learn


Downloading the Dataset
----
[Kaggle - online community of data scientists and machine learning practitioners ](https://www.kaggle.com/)