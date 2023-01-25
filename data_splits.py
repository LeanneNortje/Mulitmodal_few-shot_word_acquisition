#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

from pathlib import Path
import json
import re
from tqdm import tqdm
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize

files = "../Datasets/spokencoco/SpokenCOCO"
##########################
novel_classes = [] 
with open('./data/novel_keywords.txt', 'r') as f:
    for keyword in f:
        novel_classes.append(' '.join(keyword.split()))
##########################
novel_word_points = {}
train_word_points = {}
val_word_points = {}
train_fn = Path(files) / 'SpokenCOCO_train.json'
val_fn = Path(files) / 'SpokenCOCO_val.json'


def load_fn(fn, word_points):
    with open(fn, 'r') as f:
        data = json.load(f)

    data = data['data']

    for entry in tqdm(data):
        image = entry['image']
        added = False
        for caption in entry['captions']:
            
            for novel_word in novel_classes:
                if re.search(novel_word, caption['text'].lower()) is not None:
                    if novel_word not in word_points:
                        word_points[novel_word] = {'count': 0, 'entries': []}
                    word_points[novel_word]['count'] += 1
                    word_points[novel_word]['entries'].append((image, caption['wav'], caption['speaker']))
                    
    return word_points

train_word_points = load_fn(train_fn, train_word_points) 
val_word_points = load_fn(val_fn, val_word_points)

for w in list(train_word_points.keys()):
    if w not in val_word_points: del train_word_points[w]
    elif val_word_points[w]['count'] < 10 or train_word_points[w]['count'] < 10: 
        del train_word_points[w]
        del val_word_points[w]

for w in list(val_word_points.keys()):
    if w not in train_word_points: del val_word_points[w]

fn = Path('./data/train.json')
with open(fn, 'w') as f:
    json.dump(train_word_points, f)
print(f'Num train classes: {len(train_word_points)}')

fn = Path('./data/val.json')
with open(fn, 'w') as f:
    json.dump(val_word_points, f)
print(f'Num val classes: {len(val_word_points)}')

key = {}
id_to_word_key = {}
for i, l in enumerate(novel_classes):
    key[l] = i
    id_to_word_key[i] = l
    print(f'{i}: {l}')

np.savez_compressed(
    Path('data/label_key.npz'),
    key=key,
    id_to_word_key=id_to_word_key
)