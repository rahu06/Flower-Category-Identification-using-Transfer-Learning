import numpy as np
import os
import cv2

dir='/content/sample_data/flower' # path of the folder having subfolders for each category of the flower
base = os.listdir(dir)

x=[] #
y=[]
count=0
for i in base:
  for j in os.listdir(dir+'/'+i):
    
    img_arr=cv2.imread(dir+'/'+i+'/'+j)
    a=j.split('.')
    if a[1]=='jpg':
      y.append(i)
      img_arr=cv2.resize(img_arr,(224,224))
      x.append(img_arr)
      

xarr=np.array(x) # converion list into numpy array
yarr=np.array(y)
xarr=xarr/255.0   # Normalization of images
yarr.reshape(-1,1)  
yarr.shape

#Importing the pretrained VGG16 neural network
from tensorflow.keras.applications.vgg16 import VGG16
IMAGE_SIZE = [224, 224]
vgg = VGG16( input_shape=[224,224,3],weights='imagenet',include_top=False)
for layer in vgg.layers:
    layer.trainable = False
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
x =Flatten()(vgg.output)
prediction = Dense(2, activation='softmax')(x)
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
from tensorflow.keras.models import Model
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

model.compile(
               loss='sparse_categorical_crossentropy',
               optimizer="adam",
               metrics=['accuracy']
)

from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
#Early stopping to avoid overfitting of model
batch_size=32

model.fit(x=xarr,y=yarr,epochs=10,batch_size=4)
