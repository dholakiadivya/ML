from django.http.response import HttpResponse
from django.shortcuts import render
import pandas as pd
import numpy as np
import string
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sklearn.preprocessing import LabelEncoder
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.ensemble import RandomForestClassifier
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from imblearn import under_sampling
# Create your views here.
def home(request):
    render(request,'home.html')

def index(request):
    return render(request, 'registration.html')
def clean_text(text):
    text = text.strip().lower()
    nopunc =[char for char in text if char not in string.punctuation and not char.isdigit()]
    nopunc=''.join(nopunc)
    list = [word for word in nopunc.split()]
    return " ".join([word for word in list])
def convert_bool(value):
    if value == 'yes':
        return 1
    else :
        return 0
def predict(request):
    df = pd.read_csv('C:/Users/dhola/Project/fake_jobs.csv',index_col=0)
    title = request.POST["title"]
    company_profile = request.POST["company_profile"]
    description = request.POST["description"]
    requirements = request.POST["requirements"]
    benefits = request.POST["benefits"]
    has_questions = request.POST["has_questions"]
    has_questions = convert_bool(has_questions)
    employment_type = request.POST["employment_type"]
    loc = request.POST["loc"]
    salary_range = request.POST["salary_range"]
    salary_range = convert_bool(salary_range)
    text = title+ ' ' + company_profile+' '+description+' '+requirements+' '+benefits
    df.fillna('',inplace=True)
    df['fraudulent'] = df['fraudulent'].astype(int)
    fraudulent = ''
    
    
    dict = {'title' : [title] , 'company_profile' : [company_profile], 'description' :  [description],'requirements' :[requirements],'benefits':[benefits],'fraudulent' : [fraudulent],'has_questions' : [has_questions], 'employment_type' : [employment_type] , 'loc' : loc, 'salary_range' : [salary_range]}
    test_df = pd.DataFrame(dict)
    test_df['text'] = clean_text(text)
    del test_df['title']
    del test_df['company_profile']
    del test_df['description']
    del test_df['requirements']
    del test_df['benefits']
    df = df.append(test_df,ignore_index=True)
    enc = LabelEncoder()
    df.loc[:,['employment_type','loc']] = df.loc[:,['employment_type','loc']].apply(enc.fit_transform)
    tf = TfidfVectorizer(max_features=2000)
    df1 = pd.DataFrame(tf.fit_transform(df['text']).toarray(),columns=tf.get_feature_names())
    df.drop(['text'],axis=1,inplace=True)
    main_df = pd.concat([df1,df],axis=1)
    y_train = main_df.iloc[:17880,-1]

    x_train = main_df.iloc[:17880,:-1]
    x_test = main_df.iloc[17880:,:-1]
    x_train = np.asarray(x_train).astype('float')
    y_train = np.asarray(y_train).astype('int')
    x_test = np.asarray(x_test).astype('float')
    sm = under_sampling.RandomUnderSampler(sampling_strategy='auto',random_state=42)
    x_train_under_sample , y_train_under_sample = sm.fit_resample(x_train,y_train)
    x_train_under_sample = np.asarray(x_train_under_sample)
    y_train_under_sample = np.asarray(y_train_under_sample)
    model = Sequential()
    model.add(Dense(units=150, activation='tanh'))
    model.add(Dense(units=100, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(
        x=x_train_under_sample,
        y=y_train_under_sample,
        epochs=35,
        shuffle=True,
        verbose=2,
        batch_size = 20,
    )
    pred = model.predict(x_test)
    pred = [int(round(x[0])) for x in pred]
    prediction = pred[0]
    if (prediction == 1):
        return render(request, 'fake.html')
    else :
        return render(request, 'real.html')