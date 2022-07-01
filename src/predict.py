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

    if answer == 0:
        print('PLASMODIUM FALCIPARUM (MALARIA)')
    elif answer == 1:
        print('SARS-COV2 (COVID-19)')
    else:
        print('VIBRIO CHOLERAE (COLERA)')
    
    return answer


predict('imgs/sars-cov2.313.tif')
predict('imgs/sars-cov2.314.tif')
predict('imgs/sars-cov2.315.tif')
predict('imgs/sars-cov2.316.tif')
predict('imgs/sars-cov2.317.tif')
predict('imgs/sars-cov2.318.tif')
predict('imgs/sars-cov2.319.tif')
predict('imgs/sars-cov2.320.tif')
predict('imgs/sars-cov2.321.tif')
predict('imgs/cholera.165.png')
predict('imgs/cholera.166.png')
predict('imgs/cholera.167.png')
predict('imgs/cholera.168.png')
predict('imgs/plasmodiumFalciparum.167.png')
predict('imgs/plasmodiumFalciparum.168.png')
predict('imgs/plasmodiumFalciparum.169.png')
predict('imgs/plasmodiumFalciparum.170.png')