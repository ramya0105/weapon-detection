from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Lambda, Activation, Flatten, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
import xml.etree.ElementTree as ET
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

main = tkinter.Tk()
main.title("Weapon Detection")
main.geometry("1300x900")

global model, classes, layer_names, output_layers, colors, filename
X = []
Y = []
bb = []
#function to normalize bounding boxes
def convert_bb(img, width, height, xmin, ymin, xmax, ymax):
    bb = []
    conv_x = (128. / width)
    conv_y = (128. / height)
    height = ymax * conv_y
    width = xmax * conv_x
    x = max(xmin * conv_x, 0)
    y = max(ymin * conv_y, 0)     
    x = x / 128
    y = y / 128
    width = width/128
    height = height/128
    return x, y, width, height

def createFRCNNModel():
    global X, Y, bb
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)#split dataset into train and test
    #define input shape
    input_img = Input(shape=(128, 128, 3))
    #create FRCNN layers with 32, 64 and 512 neurons or data filteration size
    x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(input_img)
    x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
    x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    #define output layer with 4 bounding box coordinate and 1 weapan class
    x = Dense(512, activation = 'relu')(x)
    x = Dense(512, activation = 'relu')(x)
    x_bb = Dense(4, name='bb')(x)
    x_class = Dense(2, activation='softmax', name='class')(x)
    #create FRCNN Model with above input details
    frcnn_model = Model([input_img], [x_bb, x_class])
    #compile the model
    frcnn_model.compile(Adam(lr=0.001), loss=['mse', 'categorical_crossentropy'], metrics=['accuracy'])
    if os.path.exists("model/frcnn_model_weights.hdf5") == False:#if model not trained then train the model
        model_check_point = ModelCheckpoint(filepath='model/frcnn_model_weights.hdf5', verbose = 1, save_best_only = True)
        hist = frcnn_model.fit(X, [bb, Y], batch_size=32, epochs=10, validation_split=0.2, callbacks=[model_check_point])
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:#if model already trained then load it
        frcnn_model.load_weights("model/frcnn_model_weights.hdf5")
    predict = frcnn_model.predict(X_test)#perform prediction on test data
    predict = np.argmax(predict[1], axis=1)
    test = np.argmax(y_test, axis=1)
    p = precision_score(test, predict,average='macro') * 100#calculate accuracy and other metrics
    r = recall_score(test, predict,average='macro') * 100
    f = f1_score(test, predict,average='macro') * 100
    a = accuracy_score(test,predict)*100    
    text.insert(END,'FRCNN Model Accuracy  : '+str(a)+"\n")
    text.insert(END,'FRCNN Model Precision : '+str(p)+"\n")
    text.insert(END,'FRCNN Model Recall    : '+str(r)+"\n")
    text.insert(END,'FRCNN Model FMeasure  : '+str(f)+"\n\n")
    text.update_idletasks()
        

def uploadDataset():
    global X, Y, BB
    filename = filedialog.askdirectory(initialdir = "Dataset/annotations")
    if os.path.exists('model/X.txt.npy'):#if dataset images already processed then load it
        X = np.load('model/X.txt.npy') #load X images data
        Y = np.load('model/Y.txt.npy') #load weapon class label                   
        bb = np.load('model/bb.txt.npy')#load bounding boxes
        Y = to_categorical(Y)
    else:
        for root, dirs, directory in os.walk('Dataset/annotations/xmls'):#if not processed images then loop all annotation files with bounidng boxes
            for j in range(len(directory)):
                tree = ET.parse('Dataset/annotations/xmls/'+directory[j])
                root = tree.getroot()
                img_name = root.find('filename').text #read name of image
                for item in root.findall('object'):
                    name = item.find('name').text #read class id
                    xmin = int(item.find('bndbox/xmin').text) #read all bounding box coordinates
                    ymin = int(item.find('bndbox/ymin').text)
                    xmax = int(item.find('bndbox/xmax').text)
                    ymax = int(item.find('bndbox/ymax').text)
                    img = cv2.imread("Dataset/images/"+img_name)#read image path from xml
                    height, width, channel = img.shape
                    img = cv2.resize(img, (128,128))#Resize image
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    x, y, width, height = convert_bb(img, width, height, xmin, ymin, xmax, ymax)#normalized bounding boxes
                    Y.append(0) #add weapon label to Y array
                    bb.append([x, y, width, height])#add bounding boxes
                    X.append(img)
        X = np.asarray(X)#convert array to numpy format
        Y = np.asarray(Y)
        bb = np.asarray(bb)
        np.save('model/X.txt',X)#save all processed images
        np.save('model/Y.txt',Y)                    
        np.save('model/bb.txt',bb)
    text.insert(END,"Dataset Loaded\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")

def loadModel():
    text.delete('1.0', END)
    global model, classes, layer_names, output_layers, colors
    model = cv2.dnn.readNet("model/frcn.checkpoints", "model/frcn_config.cfg")
    classes = ['Weapon']
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    createFRCNNModel()
    text.insert(END,"Weapon Detection Model Loaded\n")
    
    

def detectWeapon():
    global model, classes, layer_names, output_layers, colors, filename
    img = cv2.imread(filename)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)
    outs = model.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    score = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            scr = np.amax(scores)
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                score.append(scr)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if indexes == 0:
        text.insert(END,"weapon detected in image\n")
    flag = 0
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            print(str(class_ids[i])+" "+str(score[i]))
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
            flag = 1
    if flag == 0:
        cv2.putText(img, "No weapon Detected", (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(0)

def uploadImage():
    global filename
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = "testImages")
    text.insert(END,filename+" loaded\n")

def detectVideoWeapon():
    global model, classes, layer_names, output_layers, colors, filename
    filename = askopenfilename(initialdir = "Videos")
    cap = cv2.VideoCapture(filename)
    while True:
        _, img = cap.read()
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        model.setInput(blob)
        outs = model.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        if indexes == 0: print("weapon detected in frame")
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)            
        
        cv2.imshow("Image", img)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break   
    cap.release()
    cv2.destroyAllWindows()


def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['class_accuracy']
    loss = data['class_loss']

    fig, axs = plt.subplots(1,2,figsize=(12, 6))
    axs[0].plot(accuracy, 'ro-', color = 'green')
    axs[0].set_title("FRCNN Accuracy Graph")
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    
    axs[1].plot(loss, 'ro-', color = 'red')
    axs[1].set_title("FRCNN Loss Graph")
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    plt.show()

    
font = ('times', 16, 'bold')
title = Label(main, text='Weapon Detection',anchor=W, justify=LEFT)
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')

uploadDatasetButton = Button(main, text="Upload Weapon Dataset", command=uploadDataset)
uploadDatasetButton.place(x=50,y=100)
uploadDatasetButton.config(font=font1)

loadButton = Button(main, text="Generate & Load Weapon Detection Model", command=loadModel)
loadButton.place(x=300,y=100)
loadButton.config(font=font1)

uploadButton = Button(main, text="Upload Image", command=uploadImage)
uploadButton.place(x=50,y=150)
uploadButton.config(font=font1)

detectButton = Button(main, text="Detect Weapon from Image", command=detectWeapon)
detectButton.place(x=50,y=200)
detectButton.config(font=font1)

videoButton = Button(main, text="Detect Weapon from Video", command=detectVideoWeapon)
videoButton.place(x=50,y=250)
videoButton.config(font=font1)

graphButton = Button(main, text="FRCNN Weapon Detection Training Accuracy-Loss Graph", command=graph)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)

text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=350)
text.config(font=font1)

main.config(bg='chocolate1')
main.mainloop()
