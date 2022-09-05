import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import sys


# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import flask

app = Flask(__name__)



#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import sys


# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import flask
# TensorFlow and tf.keras

from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np




# Som
#Initialize the flask App
app = Flask(__name__)
from keras.models import load_model
model=load_model("vgg32x32best500batch.h5")


from PIL import Image

def model_predict(img, model,final_features):
    
    #im = Image.open(img)
    #img = img.resize((32, 32))

    # Preprocessing the image
    #x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
   # x = np.expand_dims(x, axis=0)
    #img=cv2.imread(img)
    result=0.0
    count=0
    for i in img:
        k=Image.open(i)
        x= np.array(k).reshape((1,32,32,3))
        print(model.predict(x))
        print(count)
        count=count+1
        result=result+model.predict(x)
    #x= np.array(img).reshape((len(img),32,32,3))
    #print("----------",type(x))
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    
    
    #preds = model.predict(x)
    result=result/len(img)
    result=result*final_features[1]*final_features[2]*final_features[3]/100
    #prediction = model.predict(np.array(x).tolist()).tolist()
    return float(result)

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features)
    result=0.0
    if request.method == 'POST' and final_features[0]==1453:
        
        print(final_features)
        files = request.files.getlist("file")
        
        #prediction = model.predict(final_features)
        
        result=model_predict(files, model,final_features)
        #print(type((result[0])))
        result=str(result)
        return render_template('index.html', prediction_text='Carbon credit of the land is :{}'.format(result))
    else:
        result="password error"
    return render_template('index.html', prediction_text=result)

@app.route('/profile.html', methods=['GET','POST']) 
def profile():
    return render_template('profile.html')
@app.route('/table.html', methods=['GET','POST']) 
def table():
    return render_template('table.html')
@app.route('/login.html', methods=['GET','POST']) 
def login():
    return render_template('login.html')
@app.route('/register.html', methods=['GET','POST']) 
def register():
    return render_template('register.html')
if __name__ == "__main__":
    app.run(debug=True)