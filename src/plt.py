from unicodedata import decimal
import matplotlib.pyplot as plt

def history_parse():
    dict_ = {}
    with open("../model/history.txt", "r") as history:
        for line in history:
            keyStr, valuesStr = line.strip().split(":")
            key = keyStr[1:-1]
            valuesStr1 = valuesStr[1:-1].split(",")
            values = [float(v) for v in valuesStr1]
            dict_[key]=values
    return dict_


history_dict = history_parse()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()