'''
This code provides you the level of coordination between each pair in the data in the following format

    user1 \t user2
    coordination vector
    ...
    ...
    ----- the number of pairs

We use 10-fold cross-validation for prediction part.
Accuracy for each fold is showed after coordination vector.
'''

import json
import re
import time
import sys
import random
import json
from sklearn import svm
reload(sys)
sys.setdefaultencoding('utf-8')

#this is the address of your json file
address = 'wikipedia.talkpages.conversations_refine.txt'
#Pred = 1 if you what to use predition function
Pred = 1
#bar is minimum number of utterances a replies to b and b replies to a
bar = 10
word = list()
ID = dict()
con = dict()
admin = dict()
cnt = 0
nine = dict()
pred = list()
lx = []
ly = []

def number(tmp):
    for i in range(len(tmp)):
        if (tmp[i] < '0' or tmp[i] > '9'): return 0
    return 1

def dictionary():
    fp = open('category.txt', 'r')
    lines = fp.readlines()
    ls = 0
    #total = 0
    for line in lines:
        arr = line.split()
        word.append(dict())
        for i in range(1, len(arr), 1):
            tmp = arr[i].replace('\n', '')
            #total += 1
            word[ls][tmp] = 1
        ls += 1
    #print total

def insert(USER, REPLY_TO, CLEAN_TEXT, TIMESTAMP_UNIXTIME, UID, RID, C):
    #global cnt
    #cnt += 1
    #print a
    reply = dict()
    reply['user'] = USER
    reply['reply_user'] = REPLY_TO
    reply['text'] = CLEAN_TEXT
    reply['time'] = TIMESTAMP_UNIXTIME
    reply['uid'] = UID
    reply['rid'] = RID
    reply['consider'] = C
    user1 = USER
    user2 = REPLY_TO
    if (user1 > user2):
        a = user1
        user1 = user2
        user2 = a
    user = user1 + '@' + user2
    if (con.has_key(user) == 0): con[user] = []
    con[user].append(reply)

def build():
    fp = open(address, 'r')
    lines = fp.readlines()
    total = 0
    #print len(lines)
    for line in lines:
        shy = json.loads(line)
        ID[shy['id']] = shy['profileName']
        try:
            if (shy['status'] == 1): admin[shy['profileName']] = 1
        except Exception,e:
            sx = 1
    for line in lines:
        shy = json.loads(line)
        if (shy['replyToId'] == -1): continue
        UID = shy['id']
        a = list()
        for i in range(8): a.append(0) 
        text = shy['reviewText']
        arr = text.split()
        for i in arr:
            tmp = i.lower()
            #print i, tmp
            for j in range(8):
                if (word[j].has_key(tmp) != 0):
                    a[j] = 1
                    #print tmp, j
        nine[UID] = a
        C = 1
        try:
            C = shy['consider']
        except Exception,e:
            C = 1
        insert(shy['profileName'], ID[shy['replyToId']], shy['reviewText'], shy['unixReviewTime'], shy['id'], shy['replyToId'], C)
        
def feature():
    num = 0
    for C in con:
        pos = C.index('@')
        user1 = C[0 : pos]
        user2 = C[pos + 1 : len(C)]
        if (user1 == user2): continue
        ans = 0
        if (admin.has_key(user2) != 0): ans = 1
        li = con[C]
        ok = 1
        fea = list()
        #if (len(li) < 10): continue
        for j in range(8):
            ua = 0
            da = 0
            ub = 0
            db = 0
            for i in li:
                if (i['consider'] == 1 and ID.has_key(i['rid']) != 0 and i['user'] == user1 and nine.has_key(i['rid']) != 0 and i['reply_user'] == user2):
                    if (nine[i['rid']][j] != 0):
                        db += 1
                        if (nine[i['uid']][j] != 0): ub += 1
                    da += 1
                    if (nine[i['uid']][j] != 0): ua += 1
            if (db == 0 or da < bar):
                ok = 0
                break
            
            P2 = 1.0 * ua / da
            P1 = 1.0 * ub / db
            P = P1 - P2
            
            ua = 0
            da = 0
            ub = 0
            db = 0
            for i in li:
                if (i['consider'] == 1 and ID.has_key(i['rid']) != 0 and i['user'] == user2 and nine.has_key(i['rid']) != 0 and i['reply_user'] == user1):
                    if (nine[i['rid']][j] != 0):
                        db += 1
                        if (nine[i['uid']][j] != 0): ub += 1
                    da += 1
                    if (nine[i['uid']][j] != 0): ua += 1
            if (db == 0 or da < bar):
                ok = 0
                break
            
            P2s = 1.0 * ua / da
            P1s = 1.0 * ub / db
            Ps = P1s - P2s
            #print P1, P2, P1s, P2s, P, Ps
            if (P >= Ps): fea.append(1)
            else: fea.append(0)
        if (ok == 0): continue
        num += 1
        print user1 + '\t' + user2
        #print user1, user2, fea, ans
        case = dict()
        case['feature'] = fea
        case['v'] = ans
        pred.append(case)
        print case
    print '-----', num
    
    
def predict(idx):
    X = []
    y = []
    x_test = []
    y_test = []
    for i in range(10):
        if (i == idx):
            x_test = lx[i]
            y_test = ly[i]
        else:
            X.extend(lx[i])
            y.extend(ly[i])
    #print len(X), len(x_test)
    model = svm.SVC(C=1, gamma=1)
    model.fit(X, y)
    model.score(X, y)
    predicted = model.predict(x_test)
    hit = 0
    n = len(x_test)
    for i in range(n):
        if (predicted[i] == y_test[i]): hit = hit + 1
    acc = 1.0 * hit / n
       
    print 'fold', idx, 'acc =', acc
    return acc

def folds():
    for i in range(10):
        lx.append([])
        ly.append([])
    tot = 0
    random.shuffle(pred)
    for i in pred:
        #value = random.uniform(0, 1)
        #where = int(value / 0.1 - 1e-5)
        where = tot % 10
        tot += 1
        lx[where].append(i['feature'])
        ly[where].append(i['v'])
    tot = 0
    for i in range(10):
        tot += predict(i)
    tot /= 10
    print 'accuracy = ', tot
    
def main():
    dictionary()
    build()    
    feature()
    if (Pred == 1): folds()
    
main()
