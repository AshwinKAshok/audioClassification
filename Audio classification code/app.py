from flask import Flask,render_template,flash, url_for, request,redirect,Blueprint
from flask import Response, make_response
import requests
import json
import sys, os
import math
import numpy as np
import pandas as pd
import wave
import librosa

from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, Input, GlobalAvgPool2D, GlobalMaxPool2D, concatenate
from keras.optimizers import Adam, SGD
import keras.backend as K

from keras.models import load_model

import pymongo
from pymongo import MongoClient
mongoClient = MongoClient('localhost',27017)
db=mongoClient['coughTracker']
user_collection=db.users

bs = 128
lr = 0.003

df = pd.read_csv('input/train_post_competition.csv')

def obtain_mfcc(file, features=40):
    y, sr = librosa.load(path+file, res_type='kaiser_fast')
    return librosa.feature.mfcc(y, sr, n_mfcc=features)

def get_mfcc(file, n_mfcc=40, padding=None):
    y, sr = librosa.load(file, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=n_mfcc)
    if padding: mfcc = np.pad(mfcc, ((0, 0), (0, max(0, padding-mfcc.shape[1]))), 'constant')
    return mfcc.astype(np.float32)

lbl2idx={'Trumpet': 0,
 'Cello': 1,
 'Knock': 2,
 'Gunshot_or_gunfire': 3,
 'Hi-hat': 4,
 'Snare_drum': 5,
 'Writing': 6,
 'Laughter': 7,
 'Fart': 8,
 'Oboe': 9,
 'Cough': 10,
 'Flute': 11,
 'Bass_drum': 12,
 'Clarinet': 13,
 'Microwave_oven': 14,
 'Burping_or_eructation': 15,
 'Harmonica': 16,
 'Double_bass': 17,
 'Shatter': 18,
 'Fireworks': 19,
 'Bark': 20,
 'Tambourine': 21,
 'Telephone': 22,
 'Keys_jangling': 23,
 'Bus': 24,
 'Cowbell': 25,
 'Meow': 26,
 'Drawer_open_or_close': 27,
 'Squeak': 28,
 'Glockenspiel': 29,
 'Tearing': 30,
 'Violin_or_fiddle': 31,
 'Finger_snapping': 32,
 'Acoustic_guitar': 33,
 'Electric_piano': 34,
 'Saxophone': 35,
 'Scissors': 36,
 'Gong': 37,
 'Computer_keyboard': 38,
 'Chime': 39,
 'Applause': 40}

n_categories = len(lbl2idx)

idx2lbl = {0: 'Trumpet',
 1: 'Cello',
 2: 'Knock',
 3: 'Gunshot_or_gunfire',
 4: 'Hi-hat',
 5: 'Snare_drum',
 6: 'Writing',
 7: 'Laughter',
 8: 'Fart',
 9: 'Oboe',
 10: 'Cough',
 11: 'Flute',
 12: 'Bass_drum',
 13: 'Clarinet',
 14: 'Microwave_oven',
 15: 'Burping_or_eructation',
 16: 'Harmonica',
 17: 'Double_bass',
 18: 'Shatter',
 19: 'Fireworks',
 20: 'Bark',
 21: 'Tambourine',
 22: 'Telephone',
 23: 'Keys_jangling',
 24: 'Bus',
 25: 'Cowbell',
 26: 'Meow',
 27: 'Drawer_open_or_close',
 28: 'Squeak',
 29: 'Glockenspiel',
 30: 'Tearing',
 31: 'Violin_or_fiddle',
 32: 'Finger_snapping',
 33: 'Acoustic_guitar',
 34: 'Electric_piano',
 35: 'Saxophone',
 36: 'Scissors',
 37: 'Gong',
 38: 'Computer_keyboard',
 39: 'Chime',
 40: 'Applause'}

app = Flask(__name__)
app.secret_key='asdasd^%$%^&asdjh%^$f^'
@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/logout')
def logout():
    resp = make_response(redirect(url_for('login')))
    resp.set_cookie('clientId','', expires=0)
    return resp

@app.route('/loginVerify',methods=['GET', 'POST'])
def loginVerify():
    clientId = request.form['clientId']
    resp = make_response(redirect(url_for('index')))
    resp.set_cookie('clientId', clientId,max_age=60*60*12)
    return resp

@app.route('/index')
def index():
    if request.cookies.get('clientId') is not None:
        clientId = request.cookies.get('clientId')
        print("Inside index")
        print(clientId)
        if user_collection.find_one({"clientId":clientId}) is not None:
            coughCount = user_collection.find_one({"clientId":clientId})["coughCount"]
            print("inside if of index")
            print(coughCount)
            if coughCount > 3 :
                return render_template('index.html', clientId=clientId, coughCount=coughCount, recommendedAction="Take tylenol and visit the doc")
            else:
                return render_template('index.html', clientId=clientId, coughCount=coughCount, recommendedAction="Nil")
        else:
            data = {"clientId":clientId,"coughCount":0}
            user_collection.insert_one(data)
            print("New client created")
            return render_template('index.html', clientId=clientId, coughCount=0, recommendedAction="Nil")
    else:
        return redirect(url_for('login'))

@app.route('/saveSound',methods=['GET', 'POST'])
def saveSound():
    data = request.data
    print("hello")
    #print(data)
    with open("test_audio.wav","wb") as fo:
        fo.write(data)
    print(request)
    return Response("{'a':'b'}", status=201, mimetype='application/json')

@app.route('/audioClassify',methods=['GET', 'POST'])
def audioClassify():
    model = load_model('best_model.h5')
    model._make_predict_function()
    n_mfcc = 40
    padding = 259
    mfcc = get_mfcc("test_audio.wav", n_mfcc, padding)[None, ..., None]
    y_ = model.predict(mfcc)
    pred = idx2lbl[np.argmax(y_)]
    print(pred)
    clientId = request.cookies.get('clientId')
    print("Inside audioClassify")
    print(clientId)
    if pred == "Cough":
        coughCount = user_collection.find_one({"clientId":clientId})["coughCount"]
        print("current coughCount")
        print(coughCount)
        print("coughCount+1")
        coughCount=coughCount+1
        print("New coughCount:")
        print(coughCount)
        user_collection.update_one({"clientId":clientId},{"$set": { "coughCount": coughCount }})
    
    flash("Sound is :"+pred)
    K.clear_session()
    os.system("rm -rvf test_audio.wav")
    #return render_template('index.html')
    return redirect(url_for('index'))


if __name__ == '__main__':
    context = ('ssl.cert', 'ssl.key')
    app.run(host='0.0.0.0',port=8124,ssl_context=context)
