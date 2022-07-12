import os
import json
from gensim.models.word2vec import Word2Vec
from util import read_code_file
from code_process import parse_token
def code2tokens(file_path):
    tokens = []
    codes = read_code_file(file_path)
    for line, code in codes.items():
        for token in code.split(' '):
            parse_token(token, tokens)
    return tokens


sentense = []
traindatas = []
dir_name = '/home/zhangxs/data/ReVeal-master/data/codexglue/train'
for file in os.listdir(dir_name):
    tokens = code2tokens(os.path.join(dir_name, file))
    tokens = list(filter(None, tokens))
    sentense.append(tokens)
    data = {
        'file_name': file,
        'code': ' '.join(tokens)
    }  
    traindatas.append(data)
with open("dataset/codexglue_train.json", 'w') as f:
    json.dump(traindatas, f)
testdatas = []    
dir_name = '/home/zhangxs/data/ReVeal-master/data/codexglue/test'
for file in os.listdir(dir_name):
    tokens = code2tokens(os.path.join(dir_name, file))
    tokens = list(filter(None, tokens))
    sentense.append(tokens)
    data = {
        'file_name': file,
        'code': ' '.join(tokens)
    }  
    testdatas.append(data)

with open("dataset/codexglue_test.json", 'w') as f:
    json.dump(testdatas, f)


model = Word2Vec(sentense,vector_size=50,min_count=1, epochs=10)
model.save('codexglue-wordvec.model')