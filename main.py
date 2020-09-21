
import re
import sys
import csv
import numpy as np
import matplotlib.pylab as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer

#np.set_printoptions(threshold=np.inf)
'''
import pandas as pd
import matplotlib as plt
import statsmodels.api as sm'''


#Hiram Rios 
'''def generate_bigrams(word):
    lower = word[:-1]
    upper = word[1:]
    bigram_gen = map(lambda l,u: l+u, lower, upper)
    for bigram in bigram_gen:
        yield bigram


'''
#def compute_frequencies(data, language):
    #data is an iterable.    Frequency counts for both Bigrams and Unigrams,  on a specific language
    
  #bigram_freqs = {a+b:0 for (a,b) in itertools.product(string.ascii_lowercasestring.ascii_lowercase)}
   # letter_freqs = {a:0 for a in string.ascii_lowercase}

    #filtered = filter(lambda x: x[1] == language, data)

    #for (name, _) in filtered:
     #   for letter in name:
      #      letter_freqs[letter] += 1
       # for bigram in generate_bigrams(name):
        #    bigram_freqs[bigram] += 1

  #  return bigram_freqs, letter_freqs


    #bigram_freqs = {a+b:0 for (a,b) in itertools.product(
     #       string.ascii_lowercase, string.ascii_lowercase)}









# will give the percision of the nationality
def percision(positives, falsePos):
  answer = (positives)/(positives + falsePos)
  return answer
# gives the recall of the nationality 
def recall(positives, falseNeg):
  answer = (positives)/(positives + falseNeg)
  return answer
# gives the accuracy of the nationality 
def accuracy(positives,negatives, falseNeg, falsePos):
  answer = (positives + negatives)/(positives + negatives + falseNeg + falsePos)
  return answer 

def is_Arabic(word):
  return True

def is_chinese(word):
  if ("chao" in word) or  ("loong" in word) or ("shen" in word) or ("ang" in word) or ("eng" in word) or ("ing" in word) :
    return True
  return False

def is_czech(word):
  return True


def is_italian(word):
  
  if("ini" in word) or ("ani" in word) or ("oni" in word) or ((word.count("i"))>=3):
    return True 
  return False 


def is_dutch(word):
  return True

def is_English(word):
  if("Th" in word) or ("to" in word) or ("ol" in word) or ("Th" in word) or ("ll"in word) or ("ur" in word) or ("in" or word) or ("ra" in word) or ("on" in word) or ("ar" in word) or ("ra" in word):
    return True
  return False

def is_French(word):
  return True

def is_german(word):
  return True

def is_greek(word):
  if ("akos" in word) or ("poulos" in word) or ("ekos" in word) or ("ikis" in word) or ("is" in word) or ("as" in word ) or ("os" in word):
    return True
  return False
def is_Irish(word):
  if("O'" in word):
    return True
  return False
def is_korean(word):
  word = word.lower()
  if("ha" in word) or ("ch" in word) or ("yo" in word) or ("yu" in word) or len(word)<=2:
    return True
  return False

def is_polish(word):
  if("ski" in word):
    return True
  return False

def is_portugese(word):
  return True
#ski incrase and decrease
#in major increase and a little decrease
# think about ts maybe don't implement
#think about ala
def is_russian(word):
  if(re.findall("ov$", word)) or (re.findall("ev$", word)) or (re.findall("sky$|ski$", word)) or (re.findall("uk$", word)) or ("iev" in word) or (re.findall("ich$",word)) or (re.findall("in$",word)) or (re.findall("off$",word)) or (re.findall("ko$", word)) or (re.findall("ts$",word)) or (re.findall("ik$",word)) or ("vil" in word) or ((re.findall("^Sh",word)) and (re.findall("an$",word))) or (re.findall("lah",word)) or (re.findall("berg$",word)) or (re.findall("^Bak",word)) or  (re.findall("^Bik",word)) or (re.findall("^Zh",word)) or ("ovo" in word) or ("olo" in word):
    return True
  return False

def is_scottish(word):
  return True

def is_vietnamese(word):
 
  return False


def is_spanish(word):
    """Naive Spanish Surname Identification"""
    word = word.lower()
    keys = "áéíóúüñ"
    for letter in word:
        if letter in keys:
            return True
    return False


def is_japanese(word):
    """Naive Japanese Surname Identification"""
    if "naka" in word:
        return True
    elif "tsu" in word:
        return True
    elif "kawa" in word:
        return True
    elif "yama" in word:
      return True
    elif "awa" in word:
      return True
    elif "ishi" in word:
      return True
    
    else:
        return False

def check_nationality(word):
    """Naive Nationality Identification

    Returns "Unknown" for nationalities that are detected as 
    other than Spanish, Italian or Japanese
    """
    #if is_spanish(word):
     #   return "Spanish"
    #if is_italian(word):
     #   return "Italian"
    #if is_japanese(word):
     #   return "Japanese"
    #if is_chinese(word):
     #   return "Chinese"
   # if is_greek(word):
    #  return "Greek"
    #if is_Irish(word):
     # return "Irish"
    #if is_korean(word):
     # return "Korean"
    #if is_polish(word):
     # return "Polish"
    
    if is_russian(word):
      return "Russian"
    if is_English(word):
      return "English"
    return "Unknown"


def smooth(arr, size):
  for e in arr:
    e += 1
    e = e*(len(arr)/len(arr)+size)
  
  

if __name__ == "__main__":
    
    #if len(sys.argv) != 3:
     #   print("Usage: python b1.py " +
      #        "surnames-dev(1).csv" + "surname" )
       # sys.exit()


    
    ret = []
    correct = []
    
    russNam = np.empty((27,0))
    col1 = ["name","a","b","c","d","e","f","g","h", "i", "j", "k", "l", "m", "n", "o","p","q","r","s","t", "u", "v","w","x","y","z"]
    russNam = np.append(russNam, np.array([col1]).transpose(), axis=1)

    names = []
    russiansur = []
    
    letters = ["a","b","c","d","e","f","g","h", "i", "j", "k", "l", "m", "n", "o","p","q","r","s","t", "u", "v","w","x","y","z"]

    #print(russNam)
    with open("Russian-and-English-dev.txt", mode="r", encoding="utf-8") as input_file, \
          open("surname", mode="w", encoding="utf-8") as output_file:
        csv_reader = csv.reader(input_file, delimiter= ",")
        #for surname in input_file:
        for surname in csv_reader:
            #surname = surname.strip("")
            correct.append(surname[1])
            names.append(surname[0])
            #appends the data into an array
            surname = surname[0]
            output_file.write(surname)
            output_file.write(",")
            out = check_nationality(surname)
            ret.append(out)
            output_file.write(check_nationality(surname))
            output_file.write("\n")
    #print(ret)
    #print(correct)
    #print(len(ret))
    #print(len(correct))
    #print(names)






    # the true positive, true negative, false positive, and false negative values will be initiated to be used to find the recall, percision, and accuracy 
    positives = 0
    falseNeg= 0
    falsePos = 0
    negatives = 0
    count = 0
    print("we will produce the recall, percision and accuracy for Russian surnames ")
    for i in range(len(correct)):

      if(correct[i]=="Russian"):

        russiansur.append(names[i])
        if(ret[i]== "Russian"):
          positives += 1
          count+=1
        elif(ret[i]=="Unknown"):
          falseNeg+=1
          count+=1
        else:
          falseNeg+=1
          count+=1
      elif(correct[i]!="Russian" and ret[i]== "Russian"):
        falsePos += 1
      elif(correct[i]!="Russian" and ret[i]!= "Russian"):
        negatives += 1

    per = percision(positives, falsePos)

    rec = recall(positives, falseNeg)

    acc = accuracy(positives,negatives,falseNeg,falsePos)

    print("The percision for Russian surname is: " + str(per) )
    print("The recall for Russian surname is: " + str(rec) )
    print("The accuracy for Russian surname is: " + str(acc))
    print(str(count) + " is the numbers of words" )
    print("\n")
    #print(russiansur)
   # print(len(russiansur))
    bi2 = []
    for k in range(len(russiansur)):    
      lower =  russiansur[k][:-1]
      upper =  russiansur[k][1:]
      bigram_gen2 = map(lambda l,u: l+u, lower, upper)
      for bigram2 in bigram_gen2:
        #print( bigram)
        bi2.append(bigram2)
        

    bigram_freq2 = {}

    for c in bi2:
      if c in bigram_freq2:
        bigram_freq2[c] += 1
      else:
        bigram_freq2[c] = 1
    print(bigram_freq2)

    print("russian sur len "+str(len(russiansur)))














   





   
    engl = []
    positives = 0
    falseNeg= 0
    falsePos = 0
    negatives = 0
    count = 0
    print("we will produce the recall, percision and accuracy for English surnames")
    for i in range(len(correct)):

      if(correct[i]=="English"):

        engl.append(names[i])
        if(ret[i]== "English"):
          
          positives += 1
          count+=1
        elif(ret[i]=="Unknown"):
          falseNeg+=1
          count+=1
        else:
          falseNeg+=1
          count+=1
      elif(correct[i]!="English" and ret[i]== "English"):
        falsePos += 1
      elif(correct[i]!="English" and ret[i]!= "English"):
        negatives += 1

    per = percision(positives, falsePos)

    rec = recall(positives, falseNeg)

    acc = accuracy(positives,negatives,falseNeg,falsePos)

    print("The percision for English surname is: " + str(per) )
    print("The recall for English surname is: " + str(rec) )
    print("The accuracy for Engish surname is: " + str(acc))
    print(str(count) + " is the numbers of words" )
    print("\n")
      #print(engl)
    #for i in range(len(engl))
      #key = engl[i]
      #engBi[str(key)]= 
      #for j in range(len(engl[i])-1):
       # engBi[engl[i]] = (engl[i][j])
       # engBi[engl[i]] = (engl[i][j+1])
    #print(engBi)'''
    engBi = {}
    bi = []
   
    for i in range(len(engl)):    
      lower =  engl[i][:-1]
      upper =  engl[i][1:]
      bigram_gen = map(lambda l,u: l+u, lower, upper)
      for bigram in bigram_gen:
        #print( bigram)
        bi.append(bigram)
       

    bigram_freq = {}

    for b in bi:
      if b in bigram_freq:
        bigram_freq[b] += 1
      else:
        bigram_freq[b] = 1
    print(bigram_freq)

    
    newArr2 = np.array(tuple(bigram_freq.values()))
    newArr3 = np.array(tuple(bigram_freq2.values()))


    #newArr2 = np.array(tuple(bigram_freq.keys()))
    #print(newArr2)
    #print(len(newArr2))

    newArr2 = np.fliplr([newArr2])[0]
    #print(newArr2)
    newArr2 = np.fliplr([newArr3])[0]


    for j in russiansur:
     words = j
     words = words.lower()
     surUni = []
     surUni.append(words)
     for k in letters:
       amount = words.count(k)
       surUni.append(amount)
     russNam = np.append(russNam, np.array([surUni]).transpose(), axis=1)
    
    #print(len(russNam[0]))
    #print(russNam)
   
    name_del = np.delete(russNam,0,0)
    name_del = np.delete(name_del,0,1)
    #print(name_del)

    newArr = np.array(name_del)
  
    #this is for the vocab size for the smoothing
    vocabSize = len(engl) 
    #print(vocabSize)
    vocabSize2 = len(russiansur)

    #x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26])#.reshape(-1,1)

    x = np.array([0]*int(len(newArr2)), dtype=int).reshape(-1,1)
    x = x.transpose()

    #print(newArr2)
    y = np.array(newArr2, dtype=int).reshape(-1,1)
    smooth(y, vocabSize2)
    y = y.transpose()

    y2 = np.array(newArr3, dtype=int).reshape(-1,1)
    smooth(y2, vocabSize)
    y2 = y2.transpose()

    '''
    y2 = np.array(newArr[:, 1], dtype=int)
    y2 = y2.transpose()

    y3 = np.array(newArr[:, 2], dtype=int)
    y3 = y3.transpose()

    y4 = np.array(newArr[:, 3], dtype=int)
    y4 = y4.transpose()'''

    plt.scatter(y, x)

    m = LinearRegression()
    m.fit(y,x)
    plt.plot(y, m.predict(y), color='red')
    #plt.show()
    plt.xlabel('Bigrams')

    
    x = np.array([0]*int(len(newArr3)), dtype=int).reshape(-1,1)
    x = x.transpose()

    plt.scatter(y2,x)
    m2 = LinearRegression()
    m2.fit(y2, x)
    plt.plot(y2, m2.predict(y2), color='green')

    '''
    plt.scatter(x, y3)
    m3 = LinearRegression()
    m3.fit(x, y3)
    plt.plot(x, m3.predict(x), color='magenta')


    plt.scatter(x, y4)
    m4 = LinearRegression()
    m4.fit(x, y4)
    plt.plot(x, m4.predict(x), color='blue')
    '''
  # we put numbers here to predict
  # model.predict(123132135)
    
    plt.show()
    
    print("end")

    
  


