# -*- coding: utf-8 -*-
import h5py
import os
import re
import json
import random
import time

from pprint import pprint
from collections import Counter

import pandas as pd
import numpy as np

import urllib
from urllib import parse
from urllib.parse import urlencode

from flask import Flask, render_template, request, jsonify

from annoy import AnnoyIndex
from gensim.models import KeyedVectors,TfidfModel

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

from email.header import Header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

from bs4 import BeautifulSoup
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span

import nltk
from nltk.corpus import wordnet
# from IPython.core.display import display, HTML

import altair as alt

import warnings
warnings.filterwarnings('ignore')

###############################################################################

def get_HTML(my_doc, ent_list):
    total_html = ''
    for tem in my_doc:
        if r'/r' in tem.text or r'/n' in tem.text or '\\r' in tem.text or '\\n' in tem.text:
            total_html += '<br>'
            
        if tem not in ent_list:
            total_html += tem.text + ' '
        else:
            total_html += "<span style='background:yellow'>" + tem.text + ' ' + "</span>"
    return total_html


def get_HTML_JD(my_doc, ent_list, cv_ent_list):
    total_html = ''
    for tem in my_doc:
        if r'/r' in tem.text or r'/n' in tem.text or '\\r' in tem.text or '\\n' in tem.text:
            total_html += '<br>'

        if tem not in ent_list:
            total_html += tem.text + ' '

        elif tem.text.lower() not in cv_ent_list:
            total_html += "<span style='background:red'>" + tem.text + ' ' + "</span>"
        else:
            total_html += "<span style='background:yellow'>" + tem.text + ' ' + "</span>"

    return total_html


ner_model = spacy.load(r'ner_model/')

SKILL_Hierarchical_relation_DF = pd.read_excel(r'relation/Hierarchical_relation_DF.xls')
SKILL_Hierarchical_relation_DF = SKILL_Hierarchical_relation_DF.drop_duplicates()

SKILL_relation_Dict = {}
for ind, row in SKILL_Hierarchical_relation_DF.iterrows():
    SKILL_relation_Dict[row['node1'].lower()] = row['node2'].lower()

print("Loading:")
print( os.getcwd() )
###############################################################################
max_word_length = 500

nlp = spacy.load("en_core_web_md")

with open(r'dataset/word2idx.json', "r") as f:
    word2idx = json.load(f)

#textCNN_model = load_model(r"model/my_JD_presetation_model.h5")
textCNN_model = load_model(r"/app/model/my_JD_presetation_model.h5")

layer_output = textCNN_model.get_layer('concatenate').output
intermediate_model = tf.keras.models.Model(inputs=textCNN_model.input,outputs=layer_output)

filename = r'word2vec/glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

word2idx = {"PAD": 0,"UNK": 1} # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]

embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 2, model.vector_size))
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    word2idx[word] = i + 2
    embeddings_matrix[i + 2] = vocab_list[i][1]
    
embeddings_matrix[1] = np.mean(embeddings_matrix, axis=0)

# Job_Category = ['Receptionist','Java Developer','Cashier','Maintenance Technician','Store Manager','Physical Therapist','Project Manager','Senior Accountant','Financial Analyst','Sales Representative']
data = pd.read_csv(r'dataset/JD_dataset_300.csv',usecols = ['Query','JD'],encoding= 'latin1')
# data = data[data['Query'].isin(Job_Category)]
# data = data.groupby('Query').head(30)
# data = data.sample(300)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

def clean_text(text):
    text = text.replace(r'\\n', ' ').replace(r'\\r', ' ').replace(r'\r', ' ').replace(r'\n', ' ')
    text = cleanr.sub(' ', text)
    
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 

    text = ' '.join(word for word in text.split()) # remove stopwors from text
    return text

def remove_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def string_process(l):
    return l.split()[:max_word_length]

def get_argmax( l ):
    return np.argmax(l)

data['JD_remove_tags'] = data['JD'].apply(remove_tags)
data['JD'] = data['JD'].apply(clean_text)
data['word_list'] = data['JD'].apply( string_process )

YY = pd.get_dummies(data['Query']).values
data['category'] = list( YY )
data['category'] = data['category'].apply( get_argmax )

def PreProcessInputData( text ):
    word_labels = []

    for sequence in text:
        len_text = len(sequence)

        ###########################################
        temp_word_labels = []
        for w in sequence:
            temp_word_labels.append( word2idx.get( str(w).lower(),1 ) )

        ###########################################
        temp_word_labels = temp_word_labels + [0] * ( max_word_length - len_text )
        word_labels.append( temp_word_labels )

    return word_labels

XX = np.array( PreProcessInputData( data['word_list'] ) )
intermediate_prediction = intermediate_model.predict( XX )

JD_Vector_List = []
for i in range(0,len(intermediate_prediction)):
    JD_Vector_List.append( intermediate_prediction[i][0][0] )
    
    
#tfidf = TfidfModel.load( r"tfidf_model/my_corpora.tfidf_model" )
tfidf = TfidfModel.load( "/app/tfidf_model/my_corpora.tfidf_model" )

def get_Sentence_Vector( wordID_list ):
    word_count_list = Counter([w for w in wordID_list if w != 0]).most_common()
    return np.mean([tem[1] * embeddings_matrix[tem[0]] for tem in tfidf[word_count_list] ] ,axis=0)

Sentence_Vector_List = [ get_Sentence_Vector(row) for row in XX ]

Vector_List = []
for index in range( len(Sentence_Vector_List) ):
    Vector_List.append( np.append( Sentence_Vector_List[index], JD_Vector_List[index] ) )
    
X_tsne = TSNE(learning_rate=100).fit_transform( Vector_List )




nb = len(Vector_List)

np.random.seed(1234)            
xb = np.array( Vector_List ).astype('float32')

f = 600
t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
for i in range(300):
    t.add_item(i, xb[i])

t.build(10)


response = []
for ind in range(len(data)):
    item = {}
    item['x'] = X_tsne[ind, 0]
    item['y'] = X_tsne[ind, 1]
    item['Category'] = data['Query'].iloc[ind]
    item['JD_remove_tags'] = data['JD_remove_tags'].iloc[ind] 
    item['JD'] = data['JD'].iloc[ind] 
    response.append( item )



def extract_Emails(doc):
    email_list = []
    for token in doc:
        if token.like_email:
            email_list.append( token.text )
    return email_list



def send_Email_Gmail(mail_address, mail_content):

    #The mail addresses and password
    sender_address = 'jobrecommendation.kylg@gmail.com'
    sender_pass = 'tnugwiscgchxewho'
    receiver_address = mail_address
    cc_mail = "baipupubai@gmail.com,chengyi.ye@kylg.org"

    #Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'Demo/PoC: CV/JD Recommendation From Job Recommendation kylg'   #The subject line
    message['Cc'] = cc_mail
    #The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))
    #Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
    session.ehlo()
    session.starttls() #enable security
    session.login(sender_address, sender_pass) #login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()


resume_tem = None
#########################################################################################
app = Flask(__name__)
@app.route("/compare/")
def compare():
    print("Compare:")
    return render_template('Skill_Point_Compare.html')


@app.route("/query/")
def query():
    print( 'Query:' )
    return render_template('CV_Post.html')

@app.route("/resume_post/")
def resume_post():
    print( 'Resume_Post:' )
    return render_template('Resume_Post.html')


@app.route("/get_simmilar/", methods=['POST'])
def get_simmilar():
    global response
    global resume_tem
    response = response[:len(data)]
    print( 'get_simmilar' )
    number = 4

    my_CV_text = str(request.form['CV'])
    print( my_CV_text )

    text_remove_tags = remove_tags(my_CV_text)
    text_clean = clean_text( my_CV_text )
    my_clean_text = PreProcessInputData( [text_clean.split()[:500]]  )

    recommendation_text = ""

    if my_clean_text != '':
        print( text_clean )
        
        vector = np.append( get_Sentence_Vector( my_clean_text[0] ), intermediate_model.predict( np.array( my_clean_text ) )[0][0][0] ) 
        nearest_List = t.get_nns_by_vector(vector,number)

        x_Total = 0
        y_Total = 0
        for ind in nearest_List:
            print( X_tsne[ind, 0] )
            print( X_tsne[ind, 1] )
            print( data['Query'].iloc[ind]  )
            print( data['JD_remove_tags'].iloc[ind]  )
            x_Total  += X_tsne[ind, 0]
            y_Total  += X_tsne[ind, 1]

            recommendation_text += data['Query'].iloc[ind]
            recommendation_text += "\n"

            recommendation_text += data['JD_remove_tags'].iloc[ind]
            recommendation_text += "\n\n"

        x_Total = x_Total / number
        y_Total = y_Total / number

        item = {}
        item['x'] = x_Total
        item['y'] = y_Total
        item['Category'] = 'Resume'
        item['JD_remove_tags'] = my_CV_text
        item['JD'] = text_clean

        resume_tem = item
        response.append( item )

        emails = extract_Emails( nlp( my_CV_text ) )

        if len(emails) > 0:
            for email in emails:
                send_Email_Gmail( email, recommendation_text )
                print( "Email was sent to ", email)
        else:
            print( "No Email in CV" )

        return render_template('Job_Match.html')


@app.route("/data/getScatterPlot")
def data_ScatterPlot():
    domain = ['Resume'] + list(set(pd.DataFrame( response )['Category']) - set('Resume'))
    range_ = ['red', 'green', 'orange','pink','yellow','purple','brown','violet','peru','cyan','blue','purple']
    print( len(domain) )
    print( len(range_) )
    
    print( len(response) )
    chart = alt.Chart( pd.DataFrame( response ) ).transform_calculate(
        url='http://localhost:5000/get_gap?'+ "CV=" + response[-1]['JD_remove_tags'] + "&JD=" + alt.datum.JD_remove_tags,
    ).mark_circle(size=50).encode(
        x='x',
        y='y',
        color = alt.Color('Category', scale=alt.Scale(domain=domain, range=range_)),
        href='url:N',
        tooltip=['Category','JD']
    ).properties(
    description='JD distribute plot',
    width=800,
    height=800).interactive()

    return chart.to_json()


@app.route("/resumer_parser/", methods=['POST','GET'])
def get_resumer_parser():
    
    if request.method == 'POST': 
        CV = str(request.form['CV'])
    else:
        CV = request.args.get('CV')
        print(CV)
        CV = urllib.parse.unquote(CV)
        
    print( 'CV:' )
    print(CV)
    
    person_list = []
    person_names = person_list
    
    def get_human_names(text):
        tokens = nltk.tokenize.word_tokenize(text)
        pos = nltk.pos_tag(tokens)
        sentt = nltk.ne_chunk(pos, binary = False)
    
        person = []
        name = ""
        for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
            for leaf in subtree.leaves():
                person.append(leaf[0])
            if len(person) > 1: #avoid grabbing lone surnames
                for part in person:
                    name += part + ' '
                if name[:-1] not in person_list:
                    person_list.append(name[:-1])
                name = ''
            person = []
    #     print (person_list)
    
    names = get_human_names(CV)
    for person in person_list:
        person_split = person.split(" ")
        for name in person_split:
            if wordnet.synsets(name) or wordnet.synsets(name.lower()):
                if(name in person):
                    person_names.remove(person)
                    break
    

    phone_list = []
    pattern = [{"ORTH": "("}, {"SHAPE": "ddd"}, {"ORTH": ")"}, {"SHAPE": "dddd"}, {"ORTH": "-", "OP": "?"}, {"SHAPE": "dddd"}]
    
    matcher = Matcher(nlp.vocab)
    matcher.add("PhoneNumber", None, pattern)
    
    doc = nlp(CV)
    matches = matcher(doc)
    
    for match_id, start, end in matches:
        span = doc[start:end]
        phone_list.append( span.text )


    emails_list = []
    pattern = [{"TEXT": {"REGEX": "[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+"}}]
    
    matcher = Matcher(nlp.vocab)
    matcher.add("Email", None, pattern)
    
    doc = nlp(CV)
    matches = matcher(doc)
    
    for match_id, start, end in matches:
        span = doc[start:end]
        emails_list.append( span.text )


    
    for tem in person_names:
        CV = CV.replace( tem, "<span style='background:yellow'>" + tem + ' ' + "</span>" )

    for tem in phone_list:
        CV = CV.replace( tem, "<span style='background:yellow'>" + tem + ' ' + "</span>" )

    for tem in emails_list:
        CV = CV.replace( tem, "<span style='background:yellow'>" + tem + ' ' + "</span>" )


    return render_template('Resume_Parser.html',Resume = CV)



@app.route("/get_gap/", methods=['POST','GET'])
def get_gap():
    
    if request.method == 'POST': 
        JD = str(request.form['JD'])
        CV = str(request.form['CV'])
    else:
        JD = request.args.get('JD')
        print(JD)
        JD = urllib.parse.unquote(JD)
        CV = request.args.get('CV')
        print(CV)
        CV = urllib.parse.unquote(CV)
        
    print( 'JD:' )
    print(JD)
    print( 'CV:' )
    print(CV)
    
    print( 'get_gap:' )    
        ## skill word in JD
    doc1 = ner_model(JD)
    JD_tem_list = []
    for chunk in doc1.ents:
        JD_tem_list.append(chunk.text.lower())
    JD_tem_list = list(set(JD_tem_list))
    JD_ent_list = [t.root for t in doc1.ents]
    print( 'skill word in JD:', JD_tem_list)

    ## skill word in CV
    doc2 = ner_model(CV)
    CV_tem_list = []
    for chunk in doc2.ents:
        CV_tem_list.append(chunk.text.lower())
    CV_tem_list = list(set(CV_tem_list))

    CV_tem_list_expand = []
    for tem in CV_tem_list:
        CV_tem_list_expand.append(str(tem))
        if SKILL_relation_Dict.get(str(tem), None) is not None:
            CV_tem_list_expand.append(SKILL_relation_Dict.get(str(tem), None))

    CV_tem_list_expand = list(set(CV_tem_list_expand))
    CV_tem_list_expand = [w.lower() for w in CV_tem_list_expand]
    print( 'skill word in CV:', CV_tem_list_expand)

    CV_ent_list = [t.root for t in doc2.ents]

    jd_html = get_HTML_JD(doc1, JD_ent_list, CV_tem_list_expand)
    cv_html = get_HTML(doc2, CV_ent_list)

    gap_tem_list = []
    for t1 in JD_tem_list:
        if t1.lower() not in CV_tem_list_expand:
            gap_tem_list.append(t1)

    print('='*120)
    print( 'gap_list', gap_tem_list)

    return render_template('Compare_Result.html',JD = jd_html, CV = cv_html)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)











