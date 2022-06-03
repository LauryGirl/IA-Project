import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model

height, length = 100, 100
model = '../model/model.h5'
weigth = '../model/weigth.h5'
cnn = load_model(model)
cnn.load_weights(weigth)

def predict(img):
    x = load_img(img, target_size = (length, height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis = 0)
    array_ = cnn.predict(x) #[[1,0,0]]
    result = array_[0]
    answer = np.argmax(result)

    print(answer)
    
    return answer

predict('plasmodiumFalciparum.8.png')