
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
#Functions:
#The following function receives a string indicating the path to follow and
#and returns the data in the file.
def openFile(s):
    file = open(s)
    data_file = file.read()
    file.close()
    return data_file
#The following function receives the data from the stop word's file, splits them
#and extends them.
def cleanStop(sw):
    csw = [ts for ts in sw.split()]
    esw = ['.', ',',';',':', '/','"', '?', '!', '¡']
    csw.extend(esw)
    return csw
#   The following function utilizes beautiful soup to obtain the methodologies
#   and put them into
def bsMeth(s):
    # Use beautiful Soup to separate the methodologies
    with open(s) as fp:
        soup = BeautifulSoup(fp, 'xml')
    all = soup.find_all('Justificacion')

    counter = 1
    justificacion = ""
    for met in soup.find_all('Justificacion'):
        [s.extract() for s in met.findAll('Justificacion')]
        ####print("*Metodologia", counter)
        ###print(met)
        justificacion = justificacion + str(met) + "\n"
        ###print("\n")
        counter += 1
    return justificacion
#   The following function receives the methodologies and separates them. It
#   creates a dictionary-matrix with the following structure:
#   Methodologies-Number-Word
def sepMeth(justificacion):
    met_matrix = {}
    counter_met = 1
    for met in justificacion:
        met_vector = {}
        counter_word = 1
        met = [d for d in met.split()]
        for word in met:
            met_vector[counter_word] = word
            counter_word += 1
        if(met_vector):
            met_vector.popitem()
        met_matrix["Justificacion" + str(counter_met)] = met_vector
        counter_met += 1
    return met_matrix

#   The following function receives the Stop Words and the Methodologies and
#   removes the stop words from the methodologies (It basically returns the
#   methodologies clean.).
def cleanMeth(sw, meth):
    cleanMeth_matrix = {}
    for m, m_vector in meth.items():
        cleanMeth_vector = {}
        for num, word in m_vector.items():
            if word not in sw:
                cleanMeth_vector[num] = word
        cleanMeth_matrix[m] = cleanMeth_vector
    return cleanMeth_matrix
#   The following function receives the clean methodologies and returns a
#   dictionary with the structure Methodologies-WordVector
def listMeth(meth):
    listMethMatrix = {}
    for m, m_vector in meth.items():
        listMethVector = []
        for num, word in m_vector.items():
            listMethVector.append(word)
        listMethMatrix[m] = listMethVector
    return listMethMatrix
#   The following function gets the frequency of each word and divides them by
#   the amount of words in the document.
def relFreq(meth):
    freqMethMatrix = {}
    for m, m_vector in meth.items():
        freqMethVector = {}
        sizeM = len(m)
        freq = nltk.FreqDist(m_vector)
        for word, frequency in freq.most_common(1000):
            if word not in freqMethVector:
                freqMethVector[word] = frequency/sizeM
        freqMethMatrix[m] = freqMethVector
    return freqMethMatrix
#   The following function opens the frequency for the most common words in
#   spanish and returns a dictionary with the following structure: Word-Freq
def mostCommon():
    f = open("frecuencia.txt", "r")
    mostCommonVector = {}
    for line in f:
        line = [d for d in line.split()]
        line[2] = line[2].replace(',', '')
        mostCommonVector[line[1]] = line[2]
    return mostCommonVector
#   The following function receives the dictionary with the methodologies and
#   returns a Dictionary-Matrix with the following form:
#   Methodologies-Word-LogFrequency
def logFreq(meth, common):
    logFreqMatrix = {}
    for m, methVector in meth.items():
        logFreqVector = {}
        for word in methVector:
            ##print("Word:", word)
            if (word in common):
                ##print("Word:", word)
                ##print("Value:", common[word])
                logFreqVector[word] = np.log(float(common[word]))
            else:
                logFreqVector[word] = 0.0
        logFreqMatrix[m] = logFreqVector
    return logFreqMatrix

#   The following function creates a vocabulary for all the words that appear in
#   justifications.
def obtainVoc(TSU, Lic, Maestria, Doctorado):
    vocabulary = []
    for m, mvec in TSU.items():
        for word in mvec:
            if word not in vocabulary:
                vocabulary.append(word)
    for m, mvec in Lic.items():
        for word in mvec:
            if word not in vocabulary:
                vocabulary.append(word)
    for m, mvec in Maestria.items():
        for word in mvec:
            if word not in vocabulary:
                vocabulary.append(word)
    for m, mvec in Doctorado.items():
        for word in mvec:
            if word not in vocabulary:
                vocabulary.append(word)
    return vocabulary
#   The following function utilizes the vocabulary to dimensionate the relative
#   frequency.
def dimRelFreq(matrix, voc):
    dimRelFreqMatrix = {}
    for j, jvec in matrix.items():
        dimRelFreqVec = {}
        for word in voc:
            if word in jvec:
                dimRelFreqVec[word] = jvec[word]
            else:
                dimRelFreqVec[word] = 0.0
        dimRelFreqMatrix[j] = dimRelFreqVec
    return dimRelFreqMatrix

#   The following function subcatenates two matrices, it subtracts one from the
#   vector from the other and return the result in the following form
#   Justification-ResultantVector
def substract(A, B):
    resultantMatrix = {}
    for (a, avec), (b, bvec) in zip(A.items(), B.items()):
        resultantVec = []
        for (wa, va), (wb, vb) in zip(avec.items(), bvec.items()):
            resultantVec.append(va-vb)
        resultantMatrix[a] = resultantVec
    return resultantMatrix
#   The following function receives two dictionary matrixes, and concatenates
#   their respective vectors row by row
def concatenate(A, B):
    resultantMatrix = []
    for(a, avec), (b, bvec) in zip(A.items(), B.items()):
        resultantVec = []
        resultantVec = avec + bvec
        #print(len(resultantVec))
        resultantMatrix.append(resultantVec)
    return resultantMatrix
#   The following function takes two matrixes A, B and returns a training matrix
#   and a classification matrix. The training matrix contains all the related
#   vectors in a single matrix.
def training(A, B):
    trainingMatrix = []
    classMatrix = []
    for av in A:
        trainingMatrix.append(av)
        classMatrix.append(-1.0)
    for bv in B:
        trainingMatrix.append(bv)
        classMatrix.append(1.0)
    return (trainingMatrix, classMatrix)
#   The following function takes fifty vectors from A, B matrixes and stores
#   them in a single matrix (This will be our training matrix). The Function
#   also creates a vector as the classification vector.
def trainingR(A, B, C, D, E, F):
    trainingMatrix = []
    classMatrix = []
    for av in A:
        trainingMatrix.append(av)
        classMatrix.append(-1.0)
    for cv in C:
        trainingMatrix.append(cv)
        classMatrix.append(-1.0)
    for ev in E:
        trainingMatrix.append(ev)
        classMatrix.append(-1.0)
    for bv in B:
        trainingMatrix.append(bv)
        classMatrix.append(1.0)
    for dv in D:
        trainingMatrix.append(dv)
        classMatrix.append(1.0)
    for fv in F:
        trainingMatrix.append(fv)
        classMatrix.append(1.0)
    return(trainingMatrix, classMatrix)
#   The following function receives two dictionaries and returns the vectors
#   in the dictionary concatenated with the other matrix and a classification
#   matrix.
def trainingL(A, B):
    tMatrix = []
    cMatrix = []
    for a, avec in A.items():
        tMatrix.append(avec)
        cMatrix.append(-1.0)
    for b, bvec in B.items():
        tMatrix.append(bvec)
        cMatrix.append(1.0)
    return (tMatrix, cMatrix)
def trainingLR(A, B, C, D, E, F):
    trainingMatrix = []
    classMatrix = []
    for a, avec in A.items():
        trainingMatrix.append(avec)
        classMatrix.append(-1.0)
    for c, cvec in C.items():
        trainingMatrix.append(cvec)
        classMatrix.append(-1.0)
    for e, evec in E.items():
        trainingMatrix.append(evec)
        classMatrix.append(-1.0)
    for b, bvec in B.items():
        trainingMatrix.append(bvec)
        classMatrix.append(1.0)
    for d, dvec in D.items():
        trainingMatrix.append(dvec)
        classMatrix.append(1.0)
    for f, fvec in F.items():
        trainingMatrix.append(fvec)
        classMatrix.append(1.0)
    return(trainingMatrix, classMatrix)



#   Open the stop words file.
sw = openFile("stopWords.txt")
#   Curate the stop words.
sw = cleanStop(sw)
##print(sw)
#################################################
#               Main                            #
#################################################
justificacionTSU = bsMeth("justificacionTSU.xml")
justificacionLic = bsMeth("justificacionLic.xml")
justificacionMaestria = bsMeth("justificacionMaestria.xml")
justificacionDoctorado = bsMeth("justificacionDoctorado.xml")
print(justificacionTSU)
print(justificacionLic)
print(justificacionMaestria)
print(justificacionDoctorado)
#Separate the justifications
justificacionTSU = [d for d in justificacionTSU.split("<Justificacion>")]
justificacionLic = [d for d in justificacionLic.split("<Justificacion>")]
justificacionMaestria = [d for d in justificacionMaestria.split("<Justificacion>")]
justificacionDoctorado = [d for d in justificacionDoctorado.split("<Justificacion>")]
TSU_matrix = sepMeth(justificacionTSU)
Lic_matrix = sepMeth(justificacionLic)
Maestria_matrix = sepMeth(justificacionMaestria)
Doctorado_matrix = sepMeth(justificacionDoctorado)
print(TSU_matrix["Justificacion3"])
print(Lic_matrix["Justificacion3"])
print(Maestria_matrix["Justificacion3"])
print(Doctorado_matrix["Justificacion3"])
print(len(TSU_matrix))
print(len(Lic_matrix))
print(len(Maestria_matrix))
print(len(Doctorado_matrix))

#We remove the stop words from the files
TSU_matrix = cleanMeth(sw, TSU_matrix)
Lic_matrix = cleanMeth(sw, Lic_matrix)
Maestria_matrix = cleanMeth(sw, Maestria_matrix)
Doctorado_matrix = cleanMeth(sw, Doctorado_matrix)
print(TSU_matrix["Justificacion3"])
print(Lic_matrix["Justificacion3"])
print(Maestria_matrix["Justificacion3"])
print(Doctorado_matrix["Justificacion3"])

#We create a Justification-VectorWord dictionary (easier to handle)
TSU_matrix = listMeth(TSU_matrix)
Lic_matrix = listMeth(Lic_matrix)
Maestria_matrix = listMeth(Maestria_matrix)
Doctorado_matrix = listMeth(Doctorado_matrix)
print(TSU_matrix["Justificacion3"])
print(Lic_matrix["Justificacion3"])
print(Maestria_matrix["Justificacion3"])
print(Doctorado_matrix["Justificacion3"])

#We obtain the relative frequency for each justification
relFreqMatrixTSU = relFreq(TSU_matrix)
relFreqMatrixLic = relFreq(Lic_matrix)
relFreqMatrixMaestria = relFreq(Maestria_matrix)
relFreqMatrixDoctorado = relFreq(Doctorado_matrix)
print(relFreqMatrixTSU["Justificacion3"])
print(relFreqMatrixLic["Justificacion3"])
print(relFreqMatrixMaestria["Justificacion3"])
print(relFreqMatrixDoctorado["Justificacion3"])

#We obtain the most common frequency provided by RAE
mostCommonVec = mostCommon()
##print(mostCommonVecTSU)
logFreqMatrixTSU = logFreq(relFreqMatrixTSU, mostCommonVec)
logFreqMatrixLic = logFreq(relFreqMatrixLic, mostCommonVec)
logFreqMatrixMaestria = logFreq(relFreqMatrixMaestria, mostCommonVec)
logFreqMatrixDoctorado = logFreq(relFreqMatrixDoctorado, mostCommonVec)
print(logFreqMatrixTSU["Justificacion3"])
print(logFreqMatrixLic["Justificacion3"])
print(logFreqMatrixMaestria["Justificacion3"])
print(logFreqMatrixDoctorado["Justificacion3"])

#We obtain the vocabulary for TSU and Lic justifications.
voc = obtainVoc(TSU_matrix, Lic_matrix, Maestria_matrix, Doctorado_matrix)
print(voc)
print(len(voc))

#We dimensionate the relative frequency matrix
dimRelFreqMatrixTSU = dimRelFreq(relFreqMatrixTSU, voc)
dimRelFreqMatrixLic = dimRelFreq(relFreqMatrixLic, voc)
dimRelFreqMatrixMaestria = dimRelFreq(relFreqMatrixMaestria, voc)
dimRelFreqMatrixDoctorado = dimRelFreq(relFreqMatrixDoctorado, voc)
print(dimRelFreqMatrixTSU["Justificacion3"])
print(dimRelFreqMatrixLic["Justificacion3"])
print(dimRelFreqMatrixMaestria["Justificacion3"])
print(dimRelFreqMatrixDoctorado["Justificacion3"])

#We dimenstionate the logarithmic frequency for the most common words
dimLogFreqMatrixTSU = dimRelFreq(logFreqMatrixTSU, voc)
dimLogFreqMatrixLic = dimRelFreq(logFreqMatrixLic, voc)
dimLogFreqMatrixMaestria = dimRelFreq(logFreqMatrixMaestria, voc)
dimLogFreqMatrixDoctorado = dimRelFreq(logFreqMatrixDoctorado, voc)
print(dimLogFreqMatrixTSU["Justificacion3"])
print(dimLogFreqMatrixLic["Justificacion3"])
print(dimLogFreqMatrixMaestria["Justificacion3"])
print(dimLogFreqMatrixDoctorado["Justificacion3"])

#   We substract the relative frequency of TSU and Lic (-1)
subTSULicRF = substract(dimRelFreqMatrixTSU, dimRelFreqMatrixLic)
subTSULicLF = substract(dimLogFreqMatrixTSU, dimLogFreqMatrixLic)
subLicMaestriaRF = substract(dimRelFreqMatrixLic, dimRelFreqMatrixMaestria)
subLicMaestriaLF = substract(dimLogFreqMatrixLic, dimLogFreqMatrixMaestria)
subMaestriaDoctoradoRF = substract(dimRelFreqMatrixMaestria, dimRelFreqMatrixDoctorado)
subMaestriaDoctoradoLF = substract(dimLogFreqMatrixMaestria, dimLogFreqMatrixDoctorado)
print(subTSULicRF["Justificacion3"])
print(subTSULicLF["Justificacion3"])
print(subLicMaestriaRF["Justificacion3"])
print(subLicMaestriaLF["Justificacion3"])
print(subMaestriaDoctoradoRF["Justificacion3"])
print(subMaestriaDoctoradoLF["Justificacion3"])
print(len(subTSULicRF))
print(len(subTSULicLF))
print(len(subLicMaestriaRF))
print(len(subLicMaestriaLF))
print(len(dimLogFreqMatrixLic))
print(len(dimLogFreqMatrixMaestria))
print(len(dimLogFreqMatrixDoctorado))
print(len(subMaestriaDoctoradoLF))

#   We substract the relative frequency of Lic and TSU (+1)
subLicTSURF = substract(dimRelFreqMatrixLic, dimRelFreqMatrixTSU)
subLicTSULF = substract(dimLogFreqMatrixLic, dimLogFreqMatrixTSU)
subMaestriaLicRF = substract(dimRelFreqMatrixMaestria, dimRelFreqMatrixLic)
subMaestriaLicLF = substract(dimLogFreqMatrixMaestria, dimLogFreqMatrixLic)
subDoctoradoMaestriaRF = substract(dimRelFreqMatrixDoctorado, dimRelFreqMatrixMaestria)
subDoctoradoMaestriaLF = substract(dimLogFreqMatrixDoctorado, dimLogFreqMatrixMaestria)
print(subLicTSURF["Justificacion3"])
print(subLicTSULF["Justificacion3"])
print(subMaestriaLicRF["Justificacion3"])
print(subMaestriaLicLF["Justificacion3"])
print(subDoctoradoMaestriaRF["Justificacion3"])
print(subDoctoradoMaestriaLF["Justificacion3"])
print(len(subLicTSURF))
print(len(subLicTSULF))
print(len(subMaestriaLicRF))
print(len(subMaestriaLicLF))
print(len(subDoctoradoMaestriaRF))
print(len(subDoctoradoMaestriaLF))

#   We concatenate TSU-Lic and Lic-TSU
conTSULic = concatenate(subTSULicRF, subTSULicLF)
conLicTSU = concatenate(subLicTSURF, subLicTSULF)
conLicMaestria = concatenate(subLicMaestriaRF, subLicMaestriaLF)
conMaestriaLic = concatenate(subMaestriaLicRF, subMaestriaLicLF)
conMaestriaDoctorado = concatenate(subMaestriaDoctoradoRF, subMaestriaDoctoradoLF)
conDoctoradoMaestria = concatenate(subDoctoradoMaestriaRF, subDoctoradoMaestriaLF)
print("conTSULic: ", conTSULic[2])
print("conLicTSU: ",conLicTSU[2])
print("conLicMaestria: ", conLicMaestria[2])
print("conMaestriaLic: ", conMaestriaLic[2])
print("conMaestriaDoctorado: ", conMaestriaDoctorado[2])
print("conDoctoradoMaestria: ", conDoctoradoMaestria[2])

#   We obtain our Training Matrix and Classifying Vector.
(tMatrixG, classVecG) = trainingR(conTSULic, conLicTSU, conLicMaestria, conMaestriaLic, conMaestriaDoctorado, conDoctoradoMaestria)
(tMatrixTL, classVecTL) = training(conTSULic, conLicTSU)
(tMatrixLM, classVecLM) = training(conLicMaestria, conMaestriaLic)
(tMatrixMD, classVecMD) = training(conMaestriaDoctorado, conDoctoradoMaestria)
#  We obtain our Training Matrix with Local vectors only
(tlMatrixG, lClassVecG) = trainingLR(subTSULicRF, subLicTSURF, subLicMaestriaRF, subMaestriaLicRF, subMaestriaDoctoradoRF, subDoctoradoMaestriaRF)
(tlMatrixTL, lClassVecTL) = trainingL(subTSULicRF, subLicTSURF)
(tlMatrixLM, lClassVecLM) = trainingL(subLicMaestriaRF, subMaestriaLicRF)
(tlMatrixMD, lClassVecMD) = trainingL(subMaestriaDoctoradoRF, subDoctoradoMaestriaRF)
#   We obtain our Training Matrix with Global Vectors only
(tgMatrixG, gClassVecG) = trainingLR(subTSULicLF, subLicTSULF, subLicMaestriaLF, subMaestriaLicLF, subMaestriaDoctoradoLF, subDoctoradoMaestriaLF)
(tgMatrixTL, gClassVecTL) = trainingL(subTSULicLF, subLicTSULF)
(tgMatrixLM, gClassVecLM) = trainingL(subLicMaestriaLF, subMaestriaLicLF)
(tgMatrixMD, gClassVecMD) = trainingL(subMaestriaDoctoradoLF, subDoctoradoMaestriaLF)


# Train the Model with general values
print("General Values:")
X_train, X_test, y_train, y_test = train_test_split(tMatrixG, classVecG, test_size = 0.20)
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
# Train the Model with TSU and General
print("TSU and Licenciatura: ")
X_train, X_test, y_train, y_test = train_test_split(tMatrixTL, classVecTL, test_size = 0.20)
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
#Train the model with Licenciatura and Maestria
print("Licenciatura and Maestria")
X_train, X_test, y_train, y_test = train_test_split(tMatrixLM, classVecLM, test_size = 0.20)
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
#Train the model with Maestria and Doctorado
print("Maestria and Doctorado")
X_train, X_test, y_train, y_test = train_test_split(tMatrixMD, classVecMD, test_size = 0.20)
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#   Train the model and see the results but only for local values
print("Local General Values:")
X_train, X_test, y_train, y_test = train_test_split(tlMatrixG, lClassVecG, test_size = 0.20)
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
#   Local for TSU and Licenciatura
print("Local TSU and Licenciatura: ")
X_train, X_test, y_train, y_test = train_test_split(tlMatrixTL, lClassVecTL, test_size = 0.20)
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
#   Local for Licenciatura and Maestria
print("Local for Licenciatura and Maestria: ")
X_train, X_test, y_train, y_test = train_test_split(tlMatrixLM, lClassVecLM, test_size = 0.20)
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
#   Local for Maestria and Doctorado
print("Local for Maestria and Doctorado: ")
X_train, X_test, y_train, y_test = train_test_split(tlMatrixMD, lClassVecMD, test_size = 0.20)
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#   Train the model and see the results but only for Global Vectors
print("Global General Values:")
X_train, X_test, y_train, y_test = train_test_split(tgMatrixG, gClassVecG, test_size = 0.20)
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
#   Local for TSU and Licenciatura
print("Global TSU and Licenciatura: ")
X_train, X_test, y_train, y_test = train_test_split(tgMatrixTL, gClassVecTL, test_size = 0.20)
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
#   Local for Licenciatura and Maestria
print("Global for Licenciatura and Maestria: ")
X_train, X_test, y_train, y_test = train_test_split(tgMatrixLM, gClassVecLM, test_size = 0.20)
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
#   Local for Maestria and Doctorado
print("Global for Maestria and Doctorado: ")
X_train, X_test, y_train, y_test = train_test_split(tgMatrixMD, gClassVecMD, test_size = 0.20)
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
#   Let's do some tests for classyfying
