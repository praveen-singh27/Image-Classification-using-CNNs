from flask import Flask, request, render_template
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import os
import numpy as np

app=Flask(__name__)


model = load_model('model/MNIST_classifier_nn_model.h5')
print("@@Model loaded")

@app.route('/')
def home():
    return render_template('index.html')




@app.route("/predict",methods=['POST'])
def predict():
    if request.method =='POST':
        file = request.files['image']
        filename= file.filename
        print("@@ input posted =",filename)
        
        file_path=os.path.join('static/user_uploaded',filename)
        file.save(file_path)
        
        pred=predict_image(image=file_path)
        
        return render_template('index.html',pred_output=pred,user_image=file_path)

def predict_image(image):
    test_image=load_img(image,target_size=(28,28),color_mode='grayscale')
    print("@@Got image for prediction")
    test_image=img_to_array(test_image)/255.0
    test_image=np.expand_dims(test_image,axis=0)

    
    prediction= model.predict(test_image)              
    
    print("@@Raw result=",prediction)
    
    pred=(np.argmax(prediction[0])).round(3)
    
    if pred==0:
        return "shirt/top"
    if pred==1:
        return "Trouser"
    if pred==2:
        return "Pullover"
    if pred==3:
        return "Dress"
    if pred==4:
        return "Coat"
    if pred==5:
        return "Sandal"
    if pred==6:
        return "Shirt"
    if pred==7:
        return "Sneaker"
    if pred==8:
        return "Bag"
    else:
        return "Ankle boot"
    
  


if __name__ == '__main__':
    app.run(debug="True")
   
