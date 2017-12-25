import matplotlib.pyplot as plt
import numpy as np
import data_import
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from scipy.stats import gaussian_kde

def my_init1(shape, dtype = None):
    weight_1 = np.load("weight_1")
    return weight_1

def my_init2(shape,dtype = None):
    weight_2 = np.load("weight_2")
    return weight_2

def my_init3(shape,dtype = None):
    weight_3 = np.load("weight_3")
    return weight_3

def prepData(filePath, test_rate):
    data = data_import.import_data(filePath, test_rate=test_rate, test = 1)
    return data

def getLayerOutput(data, layer, HISorHS, afterCombTrain):
    X_train = data[0]['X_train']
    y_train = data[1]['y_train']
    
    #Model----------------
    if afterCombTrain == False:
        mlp = Sequential()
        mlp.add(Dropout(0,input_shape=(46,)))
        mlp.add(Dense(40,activation = 'relu',kernel_initializer = my_init1,activity_regularizer=regularizers.l1(10e-7),name='layer1'))
        mlp.add(Dropout(0.2))
        mlp.add(Dense(30,activation ='relu',kernel_initializer = my_init2,activity_regularizer=regularizers.l1(10e-7), name='layer2'))
        mlp.add(Dropout(0.2))
        mlp.add(Dense(20,activation = 'relu',kernel_initializer = my_init3,activity_regularizer=regularizers.l1(10e-7), name='layer3'))
        mlp.add(Dense(1,activation = 'sigmoid'))
        mlp.compile(loss = 'binary_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy'])
    else:
        mlp = Sequential()
        mlp.add(Dense(40,input_shape = (46,),activation = 'relu',name='layer1'))
        mlp.add(Dense(30,activation ='relu',name='layer2'))
        mlp.add(Dense(20,activation = 'relu',name='layer3'))
        mlp.add(Dense(1,activation = 'sigmoid'))
        mlp.load_weights("SAE_weights.h5")
    #--------------------

    layer_model = Model(inputs=mlp.input, output=mlp.get_layer(layer).output)
    layerOut = layer_model.predict(X_train)
    return layerOut

def graph(results, data, layer, HISorHS, afterCombTrain, saveToPath, analysis):
    X_train = data[0]['X_train']
    y_train = data[1]['y_train']
    dataPosEmpt = True
    dataNegEmpt = True

    for i in xrange(y_train.size):
        if y_train[i]==1:
            if dataPosEmpt == True:
                dataPos = results[i]
                dataPosEmpt = False
            else:
                dataPos = np.vstack((dataPos, results[i]))
        else:
            if dataNegEmpt == True:
                dataNeg = results[i]
                dataNegEmpt = False
            else:
                    dataNeg = np.vstack((dataNeg, results[i]))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

    colors = ("red", "blue")
    groups = ("Positive", "Negative")
    data = (dataPos, dataNeg)

    for data, color, group in zip(data, colors, groups):
        ax.scatter(data[:,0], data[:,1], s=15, alpha=0.5, color=color, label=group)
    if afterCombTrain == False:
        plt.title(analysis + " of " + layer + " output before supervised training, " + HISorHS)
    else:
        plt.title(analysis + " of " + layer + " output after supervised training, " + HISorHS)

    plt.legend(loc=2)
    plt.savefig(saveToPath, dpi=1000)

def graphDensity(results, data, layer, HISorHS, afterCombTrain, saveToPath):
    X_train = data[0]['X_train']
    y_train = data[1]['y_train']
    dataPosEmpt = True
    dataNegEmpt = True
    
    for i in xrange(y_train.size):
        if y_train[i]==1:
            if dataPosEmpt == True:
                dataPos = results[i]
                dataPosEmpt = False
            else:
                dataPos = np.vstack((dataPos, results[i]))
        else:
            if dataNegEmpt == True:
                dataNeg = results[i]
                dataNegEmpt = False
            else:
                dataNeg = np.vstack((dataNeg, results[i]))
    dataPosX = dataPos[:,0]
    dataNegX = dataNeg[:,0]

    kdePos = gaussian_kde(dataPosX)
    kdeNeg = gaussian_kde(dataNegX)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1, axisbg="1.0")

    xaxisPos = np.linspace(np.amin(dataPos[:,0])-1, np.amax(dataPos[:,0]), dataPos[:,0].size)
    xaxisNeg = np.linspace(np.amin(dataNeg[:,0])-1, np.amax(dataNeg[:,0]), dataNeg[:,0].size)

    plt.plot(xaxisPos, kdePos(xaxisPos), color='red', label='Positive')
    plt.plot(xaxisNeg, kdeNeg(xaxisNeg), color='blue', label='Negative')
    plt.ylabel("Density")
    if afterCombTrain == False:
        plt.title("Density of " + layer + " output before supervised training, " + HISorHS)
    else:
        plt.title("Density of " + layer + " output after supervised training, " + HISorHS)
    plt.legend(loc=2)
    plt.savefig(saveToPath, dpi=1000)


def PCAanalys(datafilePath, layer, HISorHS, afterCombTrain, saveFilePath):
    data = prepData(datafilePath, 0.2)
    layerOutput = getLayerOutput(data, layer, HISorHS, afterCombTrain)
    layerOut_std = StandardScaler().fit_transform(layerOutput)
    pca = PCA(n_components = 2)
    results = pca.fit_transform(layerOut_std)
    graph(results, data, layer, HISorHS, afterCombTrain, saveFilePath, "PCA")
    graphDensity(results, data, layer, HISorHS, afterCombTrain, saveFilePath+"density")

def tSNEanalys(datafilePath, layer, HISorHS, afterCombTrain, saveFilePath):
    data = prepData(datafilePath, 0.94)
    layerOut = getLayerOutput(data, layer, HISorHS, afterCombTrain)
    tsne = TSNE(n_components=2, n_iter=5000, perplexity=50)
    results = tsne.fit_transform(layerOut)
    graph(results, data, layer, HISorHS, afterCombTrain, saveFilePath, "t-SNE")

def MDSanalys(datafilePath, layer, HISorHS, afterCombTrain, saveFilePath):
    data = prepData(datafilePath, 0.98)
    layerOut = getLayerOutput(data, layer, HISorHS, afterCombTrain)
    mds = MDS(n_components=2)
    results = mds.fit_transform(layerOut.astype(np.float64))
    graph(results, data, layer, HISorHS, afterCombTrain, saveFilePath, "MDS")



