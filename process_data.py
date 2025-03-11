from ucimlrepo import fetch_ucirepo 
import numpy as np 
import pandas as pd 
import pickle


def save_data_1():
    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45) 
  
    # data (as pandas dataframes) 
    X = heart_disease.data.features.to_numpy() 
    y = heart_disease.data.targets.to_numpy()

    transformed_data = []
    for i, x in enumerate(X):
        if not np.any(np.isnan(x)):
            x = np.append(x, y[i])
            transformed_data.append(x)
    transformed_data = np.array(transformed_data)
    print(transformed_data)
    with open("data1.pickle", 'wb') as f:
       pickle.dump((transformed_data), f)

def get_data_1():
    with open("data/data1.pickle", 'rb') as f:
        data = pickle.load(f)
    
    np.random.shuffle(data)
    X = []
    y = []
    for el in data:
        X.append(el[:-1])
        y.append([int(el[-1])])

    x_train = np.array(X[:200])
    y_train = np.array(y[:200])
    x_valid = np.array(X[200:225])
    y_valid = np.array(y[200:225])
    x_test = np.array(X[225:250])
    y_test = np.array(y[225:250])

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def cross_validation_1():
    with open("data/data1.pickle", 'rb') as f:
        data = pickle.load(f)
    
    np.random.shuffle(data)
    X = []
    y = []
    for el in data:
        X.append(el[:-1])
        y.append([int(el[-1])])

    x_data = []
    y_data = []
    i_last = 0
    for i in range(25, 251, 25):
        x_data.append(X[i_last:i])
        y_data.append(y[i_last:i])
        i_last = i
    
    return (x_data, y_data)


def save_data_2():
    adult = fetch_ucirepo(id=2) 

    X = adult.data.features.to_numpy() 
    y = adult.data.targets.to_numpy()

    workclass = np.array(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
    education = np.array(['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
    marital_status = np.array(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    occupation = np.array(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    relationship = np.array(['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    race = np.array(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    gender = np.array(['Female', 'Male'])
    country = np.array(['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])
    workclass_mapping = {category: idx for idx, category in enumerate(workclass)}
    education_mapping = {category: idx for idx, category in enumerate(education)}
    marital_status_mapping = {category: idx for idx, category in enumerate(marital_status)}
    occupation_mapping = {category: idx for idx, category in enumerate(occupation)}
    relationship_mapping = {category: idx for idx, category in enumerate(relationship)}
    race_mapping = {category: idx for idx, category in enumerate(race)}
    gender_mapping = {category: idx for idx, category in enumerate(gender)}
    country_mapping = {category: idx for idx, category in enumerate(country)}

    transformed_data = []
    holder = []
    for i, x in enumerate(X):
        holder.append([x[0], workclass_mapping.get(x[1], None), x[2], education_mapping.get(x[3], None), x[4],  marital_status_mapping.get(x[5], None),  occupation_mapping.get(x[6], None),  relationship_mapping.get(x[7], None), race_mapping.get(x[8], None),  gender_mapping.get(x[9], None), x[10],  x[11], x[12], country_mapping.get(x[13], None)])
        if not None in holder[0]:
            if y[i][0] == '<=50K.' or y[i][0] == '<=50K':
                holder[0].append(0)
            elif y[i][0] == '>50K.' or y[i][0] == '>50K':
                holder[0].append(1)
            transformed_data.append(holder[0])
        holder = []
    transformed_data = np.array(transformed_data)

    with open("data2.pickle", 'wb') as f:
       pickle.dump((transformed_data), f)

def get_data_2():
    with open("data/data2.pickle", 'rb') as f:
        data = pickle.load(f)
    
    np.random.shuffle(data)
    X = []
    y = []
    for el in data[::60]:
        X.append(el[:-1])
        y.append([int(el[-1])])

    x_train = np.array(X[:400])
    y_train = np.array(y[:400])
    x_valid = np.array(X[400:450])
    y_valid = np.array(y[400:450])
    x_test = np.array(X[450:500])
    y_test = np.array(y[450:500])

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def cross_validation_2():
    with open("data/data2.pickle", 'rb') as f:
        data = pickle.load(f)
    
    np.random.shuffle(data)
    X = []
    y = []
    for el in data:
        X.append(el[:-1])
        y.append([int(el[-1])])

    x_data = []
    y_data = []
    i_last = 0
    for i in range(100, 501, 100):
        x_data.append(X[i_last:i])
        y_data.append(y[i_last:i])
        i_last = i
    
    return (x_data, y_data)


def save_data_3():
    bank_marketing = fetch_ucirepo(id=222) 

     # data (as pandas dataframes) 
    X = bank_marketing.data.features.to_numpy() 
    y = bank_marketing.data.targets.to_numpy()

    jobclass = np.array(["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"])
    martialclass = np.array(["married","divorced","single"])
    educationclass = np.array(["unknown","secondary","primary","tertiary"])
    defaultclass = np.array(["no", "yes"])
    hosuingclass = np.array(["no", "yes"])
    loanclass = np.array(["no", "yes"])
    contactclass = np.array(["unknown","telephone","cellular"])
    monthclass = np.array(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "nov", "dec"])
    poutcomeclass = np.array(["unknown","other","failure","success"])

    jobclass_mapping = {category: idx for idx, category in enumerate(jobclass)}
    martialclass_mapping = {category: idx for idx, category in enumerate(martialclass)}
    educationclass_mapping ={category: idx for idx, category in enumerate(educationclass)}
    defaultclass_mapping = {category: idx for idx, category in enumerate(defaultclass)}
    hosuingclass_mapping = {category: idx for idx, category in enumerate(hosuingclass)}
    loanclass_mapping = {category: idx for idx, category in enumerate(loanclass)}
    contactclass_mapping = {category: idx for idx, category in enumerate(contactclass)}
    monthclass_mapping = {category: idx for idx, category in enumerate(monthclass)}
    poutcomeclass_mapping = {category: idx for idx, category in enumerate(poutcomeclass)}

    transformed_data = []
    holder = []
    for i, x in enumerate(X):
        holder.append([x[0], jobclass_mapping.get(x[1], None), martialclass_mapping.get(x[2], None), educationclass_mapping.get(x[3], None), defaultclass_mapping.get(x[4], None),  x[5],  hosuingclass_mapping.get(x[6], None),  loanclass_mapping.get(x[7], None), contactclass_mapping.get(x[8], None), x[9], monthclass_mapping.get(x[10], None), x[11], x[12], x[13], x[14], poutcomeclass_mapping.get(x[15], None)])
        if not None in holder[0]:
            if y[i][0] == "no":
                holder[0].append(0)
            elif y[i][0] == "yes":
                holder[0].append(1)
            transformed_data.append(holder[0])
        holder = []
    transformed_data = np.array(transformed_data)
    with open("data3.pickle", 'wb') as f:
        pickle.dump(transformed_data, f)

def get_data_3():
    with open("data/data3.pickle", 'rb') as f:
        data = pickle.load(f)
    
    np.random.shuffle(data)
    X = []
    y = []
    for el in data[::10]:
        X.append(el[:-1])
        y.append([int(el[-1])])

    x_train = np.array(X[:400])
    y_train = np.array(y[:400])
    x_valid = np.array(X[400:450])
    y_valid = np.array(y[400:450])
    x_test = np.array(X[450:500])
    y_test = np.array(y[450:500])

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def cross_validation_3():
    with open("data/data3.pickle", 'rb') as f:
        data = pickle.load(f)
    
    np.random.shuffle(data)
    X = []
    y = []
    for el in data:
        X.append(el[:-1])
        y.append([int(el[-1])])

    x_data = []
    y_data = []
    i_last = 0
    for i in range(100, 501, 100):
        x_data.append(X[i_last:i])
        y_data.append(y[i_last:i])
        i_last = i
    
    return (x_data, y_data)


if __name__ == "__main__":
    #save_data_3()
    pass


