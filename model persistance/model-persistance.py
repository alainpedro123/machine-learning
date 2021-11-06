# MODEL PERSISTANCE - for large amount of Dataset, we build and train a model and save it to a file

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