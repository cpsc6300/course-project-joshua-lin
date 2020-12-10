

# TITLE: Predicting Who Survived: A Predictive Model for the Titanic
## Team Members: Joshua Lin
## Date: 12/10/2020

# Probelm Statement and Motivation
This project aims to see whether the prejudices or advantages that certain groups in the 1900s had were the main reasons certain groups survived the Titanic. To achieve this, we will be using a dataset of a portion of the passengers on the Titanic. The data of these passengers will involve their age, gender, social economic class, and whether they survived. We will create a model using this dataset which will predict which groups of people generally survived the Titanic.

# Introduction and Description of Data
During the early 1900s, one of the most famous shipweaks occured: the sinking of the Titanic. During this time, the Titanic did not have enough lifeboats for everyone to board, resulting in about 3/4 of the 2224 passengers and crew to die. While luck was definitely involved for some's survival, some groups definitely had some advantages which led to more of them to survive than others. This project aims to develop a model that predicts which groups of people survived along with which traits matter the most. These traits can help reveal whether the prejudices or advantages that certain groups in the 1900s had were the main reasons certain groups survived the Titanic. The data contains passengers and their traits (such as number of family members on board, age, fare, etc) along with whether they survived.

# Literature Review/Related Work 
The main vision for this project came from https://www.kaggle.com/c/titanic/overview. The dataset was also provided by Kaggle at https://www.kaggle.com/c/titanic/data. Finally, many inspirations and an overview of other's work came from https://www.kaggle.com/c/titanic/notebooks. In particular, two related work helped me plan out how I was going to complete this project: https://www.kaggle.com/startupsci/titanic-data-science-solutions and https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy.

# Modeling Approach
1. To begin, we downloaded and used the train and test data provided by Kaggle at https://www.kaggle.com/c/titanic/data and imported all relevant libraries needed for this project.

2. Next we explored the data by looking at various different summaries of the data:
    ```
    train
    train.info()
    train.describe()
    train.describe(include=['O'])
    ```

3. After exploring, we found that there were several areas we needed to clean in the data.
  a. First we dropped the PassengerID, Cabin, and Ticket columns due to them being almost unique to every single passenger, and cabin missing almost 75% of its data:
    ```
    train.drop(['PassengerId','Cabin', 'Ticket'], axis=1, inplace = True)
    test.drop(['PassengerId','Cabin', 'Ticket'], axis=1, inplace = True)
    ```
  
  b. Next we filled in the missing data that we observed in the exploration section for Age, Embarked, and Fare by using the median for the first and last, and the mode for Embarked:
  
    ```
    #Replacing missing age values
    data['Age'].fillna(data['Age'].median(), inplace = True)
    
    #Replacing missing embarked values
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)
    
    #Replacing missing fare values
    data['Fare'].fillna(data['Fare'].median(), inplace = True)
    ```
   c. There were two similar columns: SibSp (number of sibilings and spouses onboard) and Parch (number of children and parents onbaord). We combined it into one column called family members to see if there would be a higher correlation. We did not drop SibSp and Parch in case they had high correlation.
   
   ```
   data['Family_Members_On_Board'] = data['SibSp'] + data['Parch'] + 1
   data['Alone'] = 0
   data.loc[data['Family_Members_On_Board'] == 1, 'Alone'] = 1
   ```
    
   d. For age and fare, they were converted to bins and then converted into numbers based on which bin they were in. We will just show age since their process was similar:
    ```
    data['Bin_Of_Age'] = pd.cut(data['Age'].astype(int), 5)
    data.loc[ data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age'] = 4
    data['Age'] = data['Age'].astype(int)
    ```
    e. Name had to be changed since it included not just everyone's name, but their title too. The format for name was LastName, Title. RestOfName. We extracted just their title to see if there would be a correlation between one's title and their survival rate. Some titles were also very rare and did not occur often, so for those titles we categorized them into one title called rare_title:
    ```
    # Extracting the titles by using split between "," and "." since every title is between a LastName, Title. RestOfName
    data['Name'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    # Extracting all the titles that are used over 10 times into a series
    titles = (data['Name'].value_counts() < 10)
    # Renaming all the rarely used titles Rare_Title since having too many outliers wouldn't be optimal for our model
    data['Name'] = data['Name'].apply(lambda x: 'Rare_Title' if titles.loc[x] == True else x)
    ```
    f. Finally, the last thing we had to do for cleaning was converting all the categorical data into numerical. This was achieved through mapping for the names (title), sex, and embarked. We will just show the sex mapping as an example since the other two were very similar:
    ```
    data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    ```
4. Now we began modeling. Since this was a classification and regression problem, we used several different relevant models to test for the best acuracy. The models we choose to test were RandomForestClassifier, DecisionTreeClassifier, KNeighborsClassifier, GaussianNB, AdaBoostClassifier, LogisticRegression, and SVC. In addition, since the train data was already provided, we spilt the train data into X_train and y_train.
```
# Training data
X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
X_test = test
```

5. We built and predicted with each model and saved all relevant information.
```
model = model.fit(X_train, y_train)
all_models[label] = model
all_y_pred[label] = model.predict(X_test)
all_accuracy[label] = round(model.score(X_train, y_train) * 100, 2)
```

6. Finally we found out that both RandomForestClassifier and DecisionTreeClassifier tied for the highest accuracy score. at 88.78 (accuracy rate may vary depending on the way you split your train data but generally these two models should have the highest accuracy). Since we wanted to look at the feature importance, we decided to use Random Forest Classifier as out top model.

# Project Trajectory, Results, and Interpretation 

Briefly summarize any changes in your project goals or implementation plans you have made.
These changes are a natural part of any project, even those that seem the most straightforward at the beginning.
The story you tell about how you arrived at your results can powerfully illustrate your process. 

Next, show your results. How well does your model and/or implementation perform? Did you meet your goals?

Finally, give some interpretation. What do your results mean? What impact will your work have?

# Conclusions and Future Work
Summarize your results, the strengths and short-comings of your results, and speculate on how you might address these short-comings if given more time.

# References:
1. Freeman, LD. “A Data Science Framework: To Achieve 99% Accuracy.” Kaggle, 11/22/2017, https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy.
2. Kaggle. “Titanic: Machine Learning from Disaster.” Kaggle, 2011, https://www.kaggle.com/c/titanic/data.
3. Kaggle. “Titanic: Machine Learning from Disaster.” Kaggle, 2011, https://www.kaggle.com/c/titanic.
4. Sehgal, Manav. “Titanic Data Science Solutions.” Kaggle, 1/29;2017, https://www.kaggle.com/startupsci/titanic-data-science-solutions.

# Support Materials
Provide a list of materials that support your project, for example, the notebooks and data sources.

# Declaration of academic integrity and responsibility

In your report, you should include a declaration of academic integrity as follows:

```
With my signature, I certify on my honor that:

The submitted work is my and my teammates' original work and not copied from the work of someone else.
Each use of existing work of others in the submitted is cited with proper reference.
Signature: ____________ Date: ______________
```

# Credit
The above project template is based on a template developed by Harvard IACS CS109 staff (see https://github.com/Harvard-IACS/2019-CS109A/tree/master/content/projects).
