from django.http.response import HttpResponse
from django.shortcuts import render
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.ensemble import RandomForestClassifier
import logging
# Create your views here.
def home(request):
    render(request,'home.html')

def index(request):
    return render(request, 'index.html')
def clean_text(text):
    return text.strip().lower()

def predict(request):
    df = pd.read_csv('C:/Users/dhola/Project/fake_jobs.csv',index_col=0)
    title = request.POST["title"]
    company_profile = request.POST["company_profile"]
    description = request.POST["description"]
    requirements = request.POST["requirements"]
    benefits = request.POST["benefits"]
    text = title+ ' ' + company_profile+' '+description+' '+requirements+' '+benefits
    
    fraudulent = 1
    dict = {'title' : [title] , 'company_profile' : [company_profile], 'description' :  [description],'requirements' :[requirements],'benefits':[benefits],'fraudulent' : fraudulent}
    test_df = pd.DataFrame(dict)
    test_df['text'] = test_df['title']+ ' ' + test_df['company_profile'] + ' ' + test_df['description'] + ' ' + test_df['requirements'] + test_df['benefits']
    del test_df['title']
    del test_df['company_profile']
    del test_df['description']
    del test_df['requirements']
    del test_df['benefits']
    df = df.append(test_df,ignore_index=True)
    df['text'] = df['text'].apply(clean_text)
    cv = TfidfVectorizer(max_features=100)
    x = cv.fit_transform(df['text'])
    df1 = pd.DataFrame(x.toarray(),columns=cv.get_feature_names())
    df.drop(['text'],axis=1,inplace=True)
    main_df = pd.concat([df1,df],axis=1)
    y_train = main_df.iloc[:17880,-1]
    y_test = main_df.iloc[17880:,-1]
    x_train = main_df.iloc[:17880,:-1]
    x_test = main_df.iloc[17880:,:-1]
    rfc = RandomForestClassifier(n_jobs = 3,oob_score=True,n_estimators=100,criterion="entropy")
    model = rfc.fit(x_train,y_train)
    pred = rfc.predict(x_test)
    prediction = pred[0] 
    if(prediction == 1):
        return render(request, 'fake.html',{'prediction' : prediction})
    else :
        return render(request, 'real.html',{'prediction' : prediction})