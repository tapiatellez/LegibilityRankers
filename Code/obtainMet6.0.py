
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import nltk
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
#Generate random integer values
from random import seed
from random import randint
import math

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
        ######print("*Metodologia", counter)
        #####print(met)
        justificacion = justificacion + str(met) + "\n"
        #####print("\n")
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
            ####print("Word:", word)
            if (word in common):
                ####print("Word:", word)
                ####print("Value:", common[word])
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
#   The following function substracts entrance by entrance two dictionary
#   vectors. It returns the result of the subtraction but in a vector, no longer
#   a dictionary.
def substractVec(A, B):
    resultantVec = []
    for (wa, va), (wb, vb) in zip(A.items(), B.items()):
        resultantVec.append(va-vb)
    return resultantVec
#   The following function receives two dictionary matrixes, and concatenates
#   their respective vectors row by row
def concatenate(A, B):
    resultantMatrix = []
    for(a, avec), (b, bvec) in zip(A.items(), B.items()):
        resultantVec = []
        resultantVec = avec + bvec
        ###print(len(resultantVec))
        resultantMatrix.append(resultantVec)
    return resultantMatrix
#   The following function receives two list-vectors and concatenates them in
#   the order A+B. It returns the concatenated vector.
def concatenateVec(A, B):
    resultantVec = A + B
    return resultantVec
#   The followign function receives two dictionary list-vector and concatenates
#   them vector-entry by vector-entry. It returns that concatenation.
def concatenateDictionaries(A, B):
    resultantMatrix = []
    for(a, avec), (b, bvec) in zip(A.items(), B.items()):
        resultantVec = []
        for (wa, va) in avec.items():
            resultantVec.append(va)
        for(wb, vb) in bvec.items():
            resultantVec.append(vb)
        resultantMatrix.append(resultantVec)
    return resultantMatrix
#   The following function receives a dictionary-dictionary and returns a
#   a dictionary-list.
def enlist(A):
    resultantMatrix = {}
    for (a, avec) in (A.items()):
        resultantVec = []
        for(w, v) in avec.items():
            resultantVec.append(v)
        resultantMatrix[a] = resultantVec
    return resultantMatrix
#   The following function receives a dictionary-vector and returns a
#   list.
def enlistVec(A):
    resultantVec = []
    for(w, v) in A.items():
        resultantVec.append(v)
    return resultantVec

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
###############################################
#           Science Contribution              #
###############################################

#   The following function receives a dimensionalized vector from one of the
#   possible classes of justifications and returns a test vector.
def obtainTest(A, B):
    # seed random number generator
    seed(1)
    # generate the random number
    r = randint(0, len(A))
    counter = 0
    testRF = {}
    testLF = {}
    for (wordA, vwordA), (wordB, vwordB) in zip(A.items(), B.items()):
        if counter == r:
            testRF = vwordA
            testLF = vwordB
            impWord = wordA
        counter += 1
    A.pop(impWord)
    B.pop(impWord)
    return (testRF, testLF)

#   The following function receives a Matrix and returns a random vector in it.
#   This vector will be use as a representative vector of the class in order
#   to classify a text file as easier or more difficult than it.
def random(A, B):
    seed(1)
    r = randint(0, len(A))
    counter = 0
    randRF = []
    randLF = []
    for (wordA, vwordA), (wordB, vwordB) in zip(A.items(), B.items()):
        if counter == r:
            randRF = vwordA
            randLF = vwordB
            impWord = wordA
        counter += 1
    A.pop(impWord)
    B.pop(impWord)
    return(randRF, randLF)
#   The following function receives a set of vectors and returns a random vector
#   from the collection.
def randomR(A):
    seed(1)
    random = randint(0, len(A))
    randVector = A[random]
    return randVector
#   The following function receives a set of matrixes and returns a list of random
#   vectors, one for each matrix.
def obtainRandomVectorsR(A, B, C, D):
    vecList = []
    vecList.append(randomR(A))
    vecList.append(randomR(B))
    vecList.append(randomR(C))
    vecList.append(randomR(D))
    return vecList
#   The following function receives eight matrixes and returns two lists: the
#   first list contains all the random RF vectors and the second list contains
#   LF vectors.
def obtainRandomVectors(A, B, C, D, E, F, G, H):
    vecListRF = [None]*4
    vecListLF = [None]*4
    (vecListRF[0], vecListLF[0]) = random(A, B)
    (vecListRF[1], vecListLF[1]) = random(C, D)
    (vecListRF[2], vecListLF[2]) = random(E, F)
    (vecListRF[3], vecListLF[3]) = random(G, H)

    return (vecListRF, vecListLF)
#   The following function receives eight matrixes two test vector related to
#   RF and LF, the general training matrix and its classyfying vector. The
#   function returns the maximum level of difficulty of the text.
def obtainGrade(lRF, lLF, testRF, testLF, tMatrix, CV):
    #Train the SVM
    clf = svm.SVC()
    clf.fit(tMatrix, CV)

    grade = 0
    goodGrade = 0
    ##print(testRF)
    for (vecRF, vecLF) in zip(lRF, lLF):
        testSubRF = substractVec(testRF, vecRF)
        testSubLF = substractVec(testLF, vecLF)
        conTest = concatenateVec(testSubRF, testSubLF)
        result = clf.predict([conTest])
        #print("Resultado: ")
        #print(result[0])
        if (result[0] == -1.0 or goodGrade>=3):
            #print("Entering results:")
            #print("Grade: ", grade)
            if(grade == 0):
                print("Your text is as good as TSU.")
            elif(grade == 1):
                print("Your text is as good as Lic.")
            elif(grade == 2):
                print("Your text is as good as Maestria.")
            elif(grade >= 3):
                print("Your text is as good as Doctorado.")
            break
        else:
            goodGrade += 1

        grade = grade + 1
#   The following function receives a list with the respective centroids, a
#   vector test in order to try and the training matrix with its respective
#   classes.
def obtainGradeR(centroidList, testRF, testLF, tMatrix, CV):
    clf = svm.SVC()
    clf.fit(tMatrix, CV)

    grade = 0
    goodGrade = 0
    #print(testRF)
    #print(len(testRF))
    testRF = enlistVec(testRF)
    testLF = enlistVec(testLF)
    conTest = concatenateVec(testRF, testLF)
    conTest = np.array(conTest)
    #print(conTest)
    #print(len(conTest))
    result = clf.predict([conTest])
    for vec in centroidList:
        vec = np.array(vec)
        #print(vec)
        #print(len(vec))
        subVec = conTest - vec
        #print(subVec)
        #print(len(subVec))
        result = clf.predict([subVec])
        #print(result)
        if (result[0] == -1.0 or goodGrade>=3):
            if(grade == 0):
                print("Your text is as good as TSU.")
            elif(grade == 1):
                print("Your text is as good as Lic.")
            elif(grade == 2):
                print("Your text is as good as Maestria.")
            elif(grade >= 3):
                print("Your text is as good as Doctorado.")
            break
        else:
            goodGrade += 1

        grade = grade + 1
#   The following function receives a test vector and a list of random vectors.
#   It returns the grade of the vector.
def obtainGradeRandom(vector, vector_list, clf):
    grade = 0
    goodGrade = 0
    for vec in vector_list:
        vector = vector - vec
        result = clf.predict([vector])
        #print("Result: ", result)
        if (result[0] == -1.0 or goodGrade>=3):
            if(grade == 0):
                #print("Your text is as good as TSU.")
                return 0
            elif(grade == 1):
                #print("Your text is as good as Lic.")
                return 1
            elif(grade == 2):
                #print("Your text is as good as Maestria.")
                return 2
            elif(grade >= 3):
                #print("Your text is as good as Doctorado.")
                return 3
            break
        else:
            goodGrade += 1

        grade = grade + 1
#   The following function receives a testMatrix, its vector classification and a
#   list of random vector. It returns the accuracy of the evaluator.
def randomEvaluator(test_matrix, classification, vector_list, clf):
    acc = 0
    for i in range(len(test_matrix)):
        grade = obtainGradeRandom(test_matrix[i], vector_list, clf)
        #print("Grade: ", grade)
        #print("Classification: ", int(classification[i]))
        if grade == int(classification[i]):
            acc += 1
    return acc/len(test_matrix)



#   The following function receives two matrixes, one related to the Relative
#   Frequency vectors and one related to the Logarithmic Frequency. It concatenates
#   them, tranforms them into a vector instead of a dictionary and returns the
#   centroid of them all.
def centroid(A, B):
    A = enlist(A)
    B = enlist(B)
    M = concatenate(A, B)
    #Transform into a numpy array
    Marray = np.array(M)
    length = len(Marray)
    centroid = np.zeros(33642)
    for vec in Marray:
        centroid = np.add(centroid, vec)
    centroid = centroid*(1/len(M[0]))
    return centroid

def centroidR(A, B):
    A = enlist(A)
    B = enlist(B)
    M = concatenate(A, B)
    length = len(M)
    centroid = np.zeros(33642)
    for i in range(length):
        for j in range(len(M[0])):
            centroid[j] = centroid[j] + M[i][j]
    for j in range(len(M[0])):
        centroid[j] = centroid[j]/(len(M[0]))
    return centroid
#   The following function receives a matrix and returns its centroid.
def centroidRR(A):
    centroid = np.zeros(33642)
    for vec in A:
        centroid = centroid + vec
    return centroid/(len(A))
#   The following function receives the test matrix, the classification vector,
#   the centroid list and the svm classifier. It returns the accuracy of the centroid
#   evaluator.
def centroidEvaluator(test_matrix, classification, centroids, clf):
    acc = 0
    for i in range(len(test_matrix)):
        grade = obtainGradeRandom(test_matrix[i], centroids, clf)
        #print("Grade: ", grade)
        #print("Classification: ", int(classification[i]))
        if grade == int(classification[i]):
            acc += 1
    return acc/len(test_matrix)
#   The following function receives four matrixes and returns a list of centroids.
#   One for each matrix.
def obtainCentroids(A, B, C, D):
    list = []
    list.append(centroidRR(A))
    list.append(centroidRR(B))
    list.append(centroidRR(C))
    list.append(centroidRR(D))
    return list
#   The following function removes the len zero vectors from the dictionary and
#   returns a dictionary.
def removeZero(GM):
    l = []
    for n, w_vec in GM.items():
        if len(w_vec) == 0:
            l.append(n)
    for e in l:
        del GM[e]
    return GM
#   The following function receives the four grade matrixes and eliminates the
#   zero length vectors from it.
def cleanVectors(A, B, C, D):
    return removeZero(A), removeZero(B), removeZero(C), removeZero(D)
#   The following function gets the size of the smallest vector in the dictionary
#   vector-word matrix.
def smallest(GM):
    smallestValue = 1000000
    for n, w_vec in GM.items():
        if smallestValue > len(w_vec):
            smallestValue = len(w_vec)
    return smallestValue
#   The following function gets the size of the biggest vector in the dictionry
#   vector-word matrix.
def biggest(GM):
    biggestValue = 0
    for n, w_vec in GM.items():
        if biggestValue < len(w_vec):
            biggestValue = len(w_vec)
    return biggestValue
#   The following function receives a grade matrix and returns the average size
#   of the vectors in it.
def average(GM):
    averageValue = 0
    allElements = len(GM)
    for n, w_vec in GM.items():
        averageValue = averageValue + len(w_vec)
    return (averageValue/allElements)
#   The following function receives a grade matrix and returns a dictionary with
#   the size of the smallest vector, the biggest vector and the average sizes
#   of the vectors.
def returnSizes(GM):
    sizes = {}
    sizes['Smallest'] = smallest(GM)
    sizes['Biggest'] = biggest(GM)
    sizes['Average'] = average(GM)
    return sizes
#   The following function receives two grade matrixes dimensionalized and concatenated
#   and returns the substraction of both of them.
def subInc(A, B):
    matrix = []
    A = np.array(A)
    B = np.array(B)
    for vecA in A:
        for vecB in B:
            matrix.append(vecA - vecB)
    return matrix
#   The following function receives two matrices dimensionalized and concatenated
#   and returns their conmutative combination in two differente matrices. The plusOne
#   and the minusOne.
def comData(A, B):
    plusOne = subInc(B, A)
    minusOne = subInc(A, B)
    return plusOne, minusOne
#   The following function receives a matrix and returns the 80 percent of the values
#   in one matrix and the other twenty in another matrix.
def eightyTwenty(A):
    m = np.array([])
    twenty = math.ceil(len(A)*(1/5))
    for i in range(twenty):
        np.append(m, A[i], 0)
        np.delete(A, i, 0)
    return A, m
#   The following function receives the plusOne and minusOne version for all the grades
#   and returns the training matrix along with its classification vector.
def allTogetherNow(POT, MOT, POL, MOL, POM, MOM):
    M = np.array([])
    y = np.array([])
    M = np.append(POT, POL, 0)
    M = np.append(M, POM, 0)
    M = np.append(M, MOT, 0)
    M = np.append(M, MOL, 0)
    M = np.append(M, MOM, 0)
    y1 = np.ones(len(POT) + len(POL) + len(POM))
    y2 = np.zeros(len(MOT) + len(MOL) + len(MOM))
    y2 = y2-1
    y = np.append(y1, y2, 0)
    return M, y

#   The following function receives a dictionary-list matrix and returns eighty
#   percent of the vectors in one matrix and the other twenty percent in another
#   matrix.
def divideEightyTwenty(M):
    twenty = {}
    eighty = {}
    counter = 0
    stop = int(len(M)/5)
    for v, vvec in M.items():
        if counter <= stop:
            twenty[v] = vvec
        else:
            eighty[v] = vvec
        counter += 1
    return twenty, eighty
#   The following function receives two matrixes the plusOne and the minusOne and
#   returns it's respective trainingMatrix with their related classification vector.
def togetherNow(A, B):
    M = np.array([])
    y = np.array([])
    M = np.append(A,B,0)
    y1 = np.ones(len(A))
    y2 = np.zeros(len(B))
    y2 = y2-1
    y = np.append(y1, y2, 0)
    return M, y
#   The following function divides our whole training set into 80 percent for
#   training and 20 percent for testing. It returns both matrixes.
def getEightyTwenty(M, y):
    print("Enter Eighty Twenty")
    np.c_[M, y]
    length = len(M)
    testMatrix = np.array([])
    print(int(length*(1/5)))
    seed(1)
    for i in range(int(length*(1/5))):
        print(i)
        random = randint(0, length)
        np.append(testMatrix, M[random])
        np.delete(M, random, 0)
    with open('traingMatrix.txt', 'wb') as f:
        for line in trainingMatrix:
            np.savetxt(f, line, fmt = '%.2f')
    return trainingMatrix, testMatrix
#   The following function receives the M matrix which corresponds to the training
#   objects and the test matrix. It first trains the SVM for later test the accuracy
#   of it five times. Finally it returns the five accuracies along with its standard
#   deviation.
def testAccuracy(M, tM):
    #Get the training Matrix just values
    M = np.array(M)
    y = M[:, len(M[0])-1]
    M = np.delete(M, len(M[0])-1, 1)
    yt = tM[:, len(tM[0])-1]
    tM = np.delete(tM, len(tM[0])-1, 1)
    # Train the svm
    clf = svm.SVC()
    clf.fit(M)
    for i in range(5):
        counter = 0
        for j in range(len(tM)):
            if clf.predict(tM[i]) == y[i]:
                counter += 1
        accuracy.append(counter/len(tM))
    return accuracy
#   The following function receives four matrixes and constructs a single matrix
#   with the vectors of all the other matrixes and a vector with their grade
#   classification.
def testEvaluatorMatrix(A, B, C, D):
    M = np.array([])
    M = np.concatenate((A, B), axis = 0)
    M = np.concatenate((M, C), axis = 0)
    M = np.concatenate((M, D), axis = 0)
    y = np.array([])
    y1 = np.zeros(len(A))
    y2 = np.zeros(len(B)) + 1
    y3 = np.zeros(len(C))  + 2
    y4 = np.zeros(len(D)) + 3
    y = np.concatenate((y1, y2), axis = 0)
    y = np.concatenate((y, y3), axis = 0)
    y = np.concatenate((y, y4), axis = 0)
    return M, y
#   Open the stop words file.
sw = openFile("stopWords.txt")
#   Curate the stop words.
sw = cleanStop(sw)
####print(sw)
#################################################
#               Main                            #
#################################################
justificacionTSU = bsMeth("justificacionTSU.xml")
justificacionLic = bsMeth("justificacionLic.xml")
justificacionMaestria = bsMeth("justificacionMaestria.xml")
justificacionDoctorado = bsMeth("justificacionDoctorado.xml")
##print(justificacionTSU)
##print(justificacionLic)
##print(justificacionMaestria)
##print(justificacionDoctorado)
#Separate the justifications
justificacionTSU = [d for d in justificacionTSU.split("<Justificacion>")]
justificacionLic = [d for d in justificacionLic.split("<Justificacion>")]
justificacionMaestria = [d for d in justificacionMaestria.split("<Justificacion>")]
justificacionDoctorado = [d for d in justificacionDoctorado.split("<Justificacion>")]
TSU_matrix = sepMeth(justificacionTSU)
Lic_matrix = sepMeth(justificacionLic)
Maestria_matrix = sepMeth(justificacionMaestria)
Doctorado_matrix = sepMeth(justificacionDoctorado)
##print(TSU_matrix["Justificacion3"])
##print(Lic_matrix["Justificacion3"])
##print(Maestria_matrix["Justificacion3"])
##print(Doctorado_matrix["Justificacion3"])
##print(len(TSU_matrix))
##print(len(Lic_matrix))
##print(len(Maestria_matrix))
##print(len(Doctorado_matrix))

TSU_matrix, Lic_matrix, Maestria_matrix, Doctorado_matrix = cleanVectors(TSU_matrix, Lic_matrix, Maestria_matrix, Doctorado_matrix)

#   Obtain the respective sizes for the justification data.
sizesTSU = returnSizes(TSU_matrix)
sizesLic = returnSizes(Lic_matrix)
sizesMaestria = returnSizes(Maestria_matrix)
sizesDoctorado = returnSizes(Doctorado_matrix)
print(sizesTSU)
print(sizesLic)
print(sizesMaestria)
print(sizesDoctorado)

#We remove the stop words from the files
TSU_matrix = cleanMeth(sw, TSU_matrix)
Lic_matrix = cleanMeth(sw, Lic_matrix)
Maestria_matrix = cleanMeth(sw, Maestria_matrix)
Doctorado_matrix = cleanMeth(sw, Doctorado_matrix)
##print(TSU_matrix["Justificacion3"])
##print(Lic_matrix["Justificacion3"])
##print(Maestria_matrix["Justificacion3"])
##print(Doctorado_matrix["Justificacion3"])

#We create a Justification-VectorWord dictionary (easier to handle)

TSU_matrix = listMeth(TSU_matrix)
Lic_matrix = listMeth(Lic_matrix)
Maestria_matrix = listMeth(Maestria_matrix)
Doctorado_matrix = listMeth(Doctorado_matrix)

#print(TSU_matrix)
# print(Lic_matrix["Justificacion3"])
# print(Maestria_matrix["Justificacion3"])
# print(Doctorado_matrix["Justificacion3"])

#Divide our data in eighty and twenty percent(in order to do tests)
twentyTSU_matrix, eightyTSU_matrix = divideEightyTwenty(TSU_matrix)
twentyLic_matrix, eightyLic_matrix = divideEightyTwenty(Lic_matrix)
twentyMaestria_matrix, eightyMaestria_matrix = divideEightyTwenty(Maestria_matrix)
twentyDoctorado_matrix, eightyDoctorado_matrix = divideEightyTwenty(Doctorado_matrix)
print("Data we are working.")
print("TSU twenty: ", len(twentyTSU_matrix))
print("TSU eighty: ", len(eightyTSU_matrix))
print("Lic twenty: ", len(twentyLic_matrix))
print("Lic eighty: ", len(eightyLic_matrix))
print("Maestria twenty: ", len(twentyMaestria_matrix))
print("Maestria eighty: ", len(eightyMaestria_matrix))
print("Doctorado twenty: ", len(twentyDoctorado_matrix))
print("Doctorado eighty: ", len(eightyDoctorado_matrix))



#We obtain the relative frequency for each justification
#First for the twenty percent of data
twentyRelFreqMatrixTSU = relFreq(twentyTSU_matrix)
twentyRelFreqMatrixLic = relFreq(twentyLic_matrix)
twentyRelFreqMatrixMaestria = relFreq(twentyMaestria_matrix)
twentyRelFreqMatrixDoctorado = relFreq(twentyDoctorado_matrix)
#Next for the eighty percent of data
eightyRelFreqMatrixTSU = relFreq(eightyTSU_matrix)
eightyRelFreqMatrixLic = relFreq(eightyLic_matrix)
eightyRelFreqMatrixMaestria = relFreq(eightyMaestria_matrix)
eightyRelFreqMatrixDoctorado = relFreq(eightyDoctorado_matrix)
##print(relFreqMatrixTSU["Justificacion3"])
##print(relFreqMatrixLic["Justificacion3"])
##print(relFreqMatrixMaestria["Justificacion3"])
##print(relFreqMatrixDoctorado["Justificacion3"])
#
#We obtain the most common frequency provided by RAE
mostCommonVec = mostCommon()
####print(mostCommonVecTSU)
twentyLogFreqMatrixTSU = logFreq(twentyRelFreqMatrixTSU, mostCommonVec)
twentyLogFreqMatrixLic = logFreq(twentyRelFreqMatrixLic, mostCommonVec)
twentyLogFreqMatrixMaestria = logFreq(twentyRelFreqMatrixMaestria, mostCommonVec)
twentyLogFreqMatrixDoctorado = logFreq(twentyRelFreqMatrixDoctorado, mostCommonVec)
#Now we obtain for the eighty percent of data
eightyLogFreqMatrixTSU = logFreq(eightyRelFreqMatrixTSU, mostCommonVec)
eightyLogFreqMatrixLic = logFreq(eightyRelFreqMatrixLic, mostCommonVec)
eightyLogFreqMatrixMaestria = logFreq(eightyRelFreqMatrixMaestria, mostCommonVec)
eightyLogFreqMatrixDoctorado = logFreq(eightyRelFreqMatrixDoctorado, mostCommonVec)
#print(logFreqMatrixTSU["Justificacion3"])
#print(logFreqMatrixLic["Justificacion3"])
#print(logFreqMatrixMaestria["Justificacion3"])
#print(logFreqMatrixDoctorado["Justificacion3"])

#We obtain the vocabulary for TSU and Lic justifications.
voc = obtainVoc(TSU_matrix, Lic_matrix, Maestria_matrix, Doctorado_matrix)
# ##print(voc)
print("The length of the vocabulary", len(voc))
#
#We dimensionate the relative frequency matrix
#For the twenty percent of the data
twentyDimRelFreqMatrixTSU = dimRelFreq(twentyRelFreqMatrixTSU, voc)
twentyDimRelFreqMatrixLic = dimRelFreq(twentyRelFreqMatrixLic, voc)
twentyDimRelFreqMatrixMaestria = dimRelFreq(twentyRelFreqMatrixMaestria, voc)
twentyDimRelFreqMatrixDoctorado = dimRelFreq(twentyRelFreqMatrixDoctorado, voc)
#For the rest of the data
eightyDimRelFreqMatrixTSU = dimRelFreq(eightyRelFreqMatrixTSU, voc)
eightyDimRelFreqMatrixLic = dimRelFreq(eightyRelFreqMatrixLic, voc)
eightyDimRelFreqMatrixMaestria = dimRelFreq(eightyRelFreqMatrixMaestria, voc)
eightyDimRelFreqMatrixDoctorado = dimRelFreq(eightyRelFreqMatrixDoctorado, voc)
# ##print(dimRelFreqMatrixTSU["Justificacion3"])
# ##print(dimRelFreqMatrixLic["Justificacion3"])
# ##print(dimRelFreqMatrixMaestria["Justificacion3"])
# ##print(dimRelFreqMatrixDoctorado["Justificacion3"])
#We dimenstionate the logarithmic frequency for the most common words
#For the twenty percent of the data
twentyDimLogFreqMatrixTSU = dimRelFreq(twentyLogFreqMatrixTSU, voc)
twentyDimLogFreqMatrixLic = dimRelFreq(twentyLogFreqMatrixLic, voc)
twentyDimLogFreqMatrixMaestria = dimRelFreq(twentyLogFreqMatrixMaestria, voc)
twentyDimLogFreqMatrixDoctorado = dimRelFreq(twentyLogFreqMatrixDoctorado, voc)
#For the eighty percent of the data
eightyDimLogFreqMatrixTSU = dimRelFreq(eightyLogFreqMatrixTSU, voc)
eightyDimLogFreqMatrixLic = dimRelFreq(eightyLogFreqMatrixLic, voc)
eightyDimLogFreqMatrixMaestria = dimRelFreq(eightyLogFreqMatrixMaestria, voc)
eightyDimLogFreqMatrixDoctorado = dimRelFreq(eightyLogFreqMatrixDoctorado, voc)

# ##print(dimLogFreqMatrixTSU["Justificacion3"])
# ##print(dimLogFreqMatrixLic["Justificacion3"])
# ##print(dimLogFreqMatrixMaestria["Justificacion3"])
# #print(dimLogFreqMatrixDoctorado["Justificacion3"])

#   We concatenate the vectors
#For the first twenty percent
twentyConTSU = concatenateDictionaries(twentyDimRelFreqMatrixTSU, twentyDimLogFreqMatrixTSU)
twentyConLic = concatenateDictionaries(twentyDimRelFreqMatrixLic, twentyDimLogFreqMatrixLic)
twentyConMaestria = concatenateDictionaries(twentyDimRelFreqMatrixMaestria, twentyDimLogFreqMatrixMaestria)
twentyConDoctorado = concatenateDictionaries(twentyDimRelFreqMatrixDoctorado, twentyDimLogFreqMatrixDoctorado)
#For the next eighty percent
eightyConTSU = concatenateDictionaries(eightyDimRelFreqMatrixTSU, eightyDimLogFreqMatrixTSU)
eightyConLic = concatenateDictionaries(eightyDimRelFreqMatrixLic, eightyDimLogFreqMatrixLic)
eightyConMaestria = concatenateDictionaries(eightyDimRelFreqMatrixMaestria, eightyDimLogFreqMatrixMaestria)
eightyConDoctorado = concatenateDictionaries(eightyDimRelFreqMatrixDoctorado, eightyDimLogFreqMatrixDoctorado)

# # print(conTSU[0])
# # print(len(conTSU))
# # print(len(conTSU[0]))
#
#  We get the +1 and -1 classification
# For the twenty percent of the data
twentyPOneTSU, twentyMOneTSU = comData(twentyConTSU, twentyConLic)
twentyPOneLic, twentyMOneLic = comData(twentyConLic, twentyConMaestria)
twentyPOneMaestria, twentyMOneMaestria = comData(twentyConMaestria, twentyConDoctorado)
# For the eighty percent of the data
eightyPOneTSU, eightyMOneTSU = comData(eightyConTSU, eightyConLic)
eightyPOneLic, eightyMOneLic = comData(eightyConLic, eightyConMaestria)
eightyPOneMaestria, eightyMOneMaestria = comData(twentyConMaestria, twentyConDoctorado)

# # print(POneTSU[0])
print("TSU-Lic twenty:", len(twentyPOneTSU))
print("Lic-TSU twenty:", len(twentyMOneTSU))
print("Lic-Maestria twenty:", len(twentyPOneLic))
print("Maestria-Lic twenty:", len(twentyMOneLic))
print("Maestria-Doctorado twenty:", len(twentyPOneMaestria))
print("Doctorado-Maestria twenty:", len(twentyMOneMaestria))
print("TSU-Lic eighty:", len(eightyPOneTSU))
print("Lic-TSU eighty:", len(eightyMOneTSU))
print("Lic-Maestria eighty:", len(eightyPOneLic))
print("Maestria-Lic eighty:", len(eightyMOneLic))
print("Maestria-Doctorado eighty:", len(eightyPOneMaestria))
print("Doctorado-Maestria eighty:", len(eightyMOneMaestria))
#   Get training matrix for a single pair of grades
# MD, yd = togetherNow(POneTSU, MOneTSU)
#Get eighty twenty for each of the training data (This for tests)
#eighty, twenty = eightyTwenty(POneTSU)

#   We put all the data together
tM, ty = allTogetherNow(twentyPOneTSU, twentyMOneTSU, twentyPOneLic, twentyMOneLic, twentyPOneMaestria, twentyMOneMaestria)
TM, Ty = allTogetherNow(eightyPOneTSU, eightyMOneTSU, eightyPOneLic, eightyMOneLic, eightyPOneMaestria, eightyMOneMaestria)

# print(M)
# print(y)
# print("Length M:", len(M))
# print("Length y:", len(y))
#
#   Train the model and see the results but only for Global Vectors
print("Global General Values:")
# X_train, X_test, y_train, y_test = train_test_split(M, y, test_size = 0.20)
#Training the classifier
svclassifier = SVC()
svclassifier.fit(tM, ty)
#for i in range(5):
#    y_pred = svclassifier.predict(TM)
    #print(confusion_matrix(Ty,y_pred))
#    print(classification_report(Ty,y_pred))


#Obtain the list with the random vectors
random_list = obtainRandomVectorsR(eightyConTSU, eightyConLic, eightyConMaestria, eightyConDoctorado)
#Obtain the matrix with the test vectors to measure our evaluator.
evaluator_matrix, y = testEvaluatorMatrix(eightyConTSU, eightyConLic, eightyConMaestria, eightyConDoctorado)
print("Length of random list vectors: ", len(random_list))
print("Length of the evaluator matrix: ", len(evaluator_matrix))
print("Length of the evaluator vector: ", len(y))
#Evaluate the matrix
acc_random = randomEvaluator(evaluator_matrix, y, random_list, svclassifier)
print("Accuracy for random evaluator: ", acc_random)


#Obtain the list with the centroids
centroid_list = obtainCentroids(eightyConTSU, eightyConLic, eightyConMaestria, eightyConDoctorado)
acc_centroids = centroidEvaluator(evaluator_matrix, y, centroid_list, svclassifier)
print("Accuracy for centroid evaluator: ", acc_centroids)
