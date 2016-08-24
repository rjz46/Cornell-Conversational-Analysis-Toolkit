import json
import time

cat_its = dict()
cat_sti = dict()
cat_fw = dict()
interest = dict()
word = list()
ID = dict()
con = dict()
admin = dict()
cnt = 0
nine = dict()
pred = list()

def number(tmp):
    for i in range(len(tmp)):
        if (tmp[i] < '0' or tmp[i] > '9'): return 0
    return 1

def datetime_timestamp(dt):
     time.strptime(dt, '%Y-%m-%d %H:%M:%S')
     s = time.mktime(time.strptime(dt, '%Y-%m-%d %H:%M:%S'))
     return int(s)
    
def calc(Time):
    Year = Time[0 : 4]
    Month = Time[5 : 7]
    Day = Time[8 : 10]
    go = Year + "-" + Month + "-" + Day + " "
    Hour = 23
    Min = '59'
    Sec = '59'
    go += (str)(Hour) + ':' + Min + ':' + Sec
    #print go, datetime_timestamp(go)
    return datetime_timestamp(go)

def insert(USER, CLEAN_TEXT, TIMESTAMP_UNIXTIME, UID, RID, fp):
    total = 0
    AD = ''
    NAD = ''
    #print USER, REPLY_TO, len(admin)
    REPLY_TO = ID[RID]
    if (RID == -1):
        shy = dict()
        shy['id'] = UID
        shy['unixReviewTime'] = TIMESTAMP_UNIXTIME
        shy['replyToId'] = RID
        #shy['reply'] = ID[RID]
        shy['profileName'] = USER
        shy['reviewText'] = CLEAN_TEXT
        shy['consider'] = 0
        if (admin.has_key(USER) != 0): shy['status'] = 1
        else: shy['status'] = 0
        fp.write(json.dumps(shy))
        fp.write('\n')
        return 0
        
    if (admin.has_key(USER) != 0): total += 1
    if (admin.has_key(REPLY_TO) != 0): total += 1
    if (total != 1):
        shy = dict()
        shy['id'] = UID
        shy['unixReviewTime'] = TIMESTAMP_UNIXTIME
        shy['replyToId'] = RID
        #shy['reply'] = ID[RID]
        shy['profileName'] = USER
        shy['reviewText'] = CLEAN_TEXT
        shy['consider'] = 0
        if (admin.has_key(USER) != 0): shy['status'] = 1
        else: shy['status'] = 0
        fp.write(json.dumps(shy))
        fp.write('\n')
        return 0
    
    if (admin.has_key(USER) != 0):
        AD = USER
        NAD = REPLY_TO
    else:
        NAD = USER
        AD = REPLY_TO
    #print 'alive'
    if (TIMESTAMP_UNIXTIME < admin[AD]):
        shy = dict()
        shy['id'] = UID
        shy['unixReviewTime'] = TIMESTAMP_UNIXTIME
        shy['replyToId'] = RID
        #shy['reply'] = ID[RID]
        shy['profileName'] = USER
        shy['reviewText'] = CLEAN_TEXT
        shy['consider'] = 0
        if (admin.has_key(USER) != 0): shy['status'] = 1
        else: shy['status'] = 0
        fp.write(json.dumps(shy))
        fp.write('\n')
        return 0
    
    
    shy = dict()
    shy['id'] = UID
    shy['unixReviewTime'] = TIMESTAMP_UNIXTIME
    shy['replyToId'] = RID
    #shy['reply'] = ID[RID]
    shy['profileName'] = USER
    shy['reviewText'] = CLEAN_TEXT
    shy['consider'] = 1
    if (admin.has_key(USER) != 0): shy['status'] = 1
    else: shy['status'] = 0
    fp.write(json.dumps(shy))
    fp.write('\n')
    
def build():
    fp = open('wikipedia.talkpages.admins.txt', 'r')
    lines = fp.readlines()
    for line in lines:
        #print line
        user = ''
        Time = '2099-12-31'
        if (line[len(line) - 2] != 'A'):
            k = line[0 : len(line) - 12]
            v = line[len(line) - 11 : len(line) - 1]
            Time = v
            user = k
        else: user = line[0 : len(line) - 4]
        #print user, len(user)
        admin[user] = calc(Time)
    print 'user', len(admin)
    
    fp = open('wikipedia.talkpages.conversations.txt', 'r')
    lines = fp.readlines()
    total = 0
    for line in lines:
        arr = line.split(' +++$+++ ')
        if (len(arr) != 9): continue
        total += 1
        UTTERANCE_ID = arr[0]
        USER = arr[1]
        TALKPAGE_USER = arr[2]
        CONVERSATION_ROOT = arr[3]
        REPLY_TO = arr[4]
        TIMESTAMP = arr[5]
        TIMESTAMP_UNIXTIME = arr[6]
        CLEAN_TEXT = arr[7]
        ID[UTTERANCE_ID] = USER
    print total

    fp = open('wikipedia.talkpages.conversations_refine.txt', 'w')
    total = 0
    ID[-1] = '!!!'
    for line in lines:
        arr = line.split(' +++$+++ ')
        if (len(arr) != 9): continue
        UTTERANCE_ID = arr[0]
        USER = arr[1]
        TALKPAGE_USER = arr[2]
        CONVERSATION_ROOT = arr[3]
        REPLY_TO = arr[4]
        TIMESTAMP = arr[5]
        TIMESTAMP_UNIXTIME = arr[6]
        CLEAN_TEXT = arr[7]
        shy = REPLY_TO
        if (number(REPLY_TO) == 0):
            shy = -1
        #print ID[-1]
        #print UTTERANCE_ID, shy, ID[shy]
        total += 1
        #print line, USER, ID[REPLY_TO]
        insert(USER, CLEAN_TEXT, TIMESTAMP_UNIXTIME, UTTERANCE_ID, shy, fp)
        #if (total > 100): return 0

build()
