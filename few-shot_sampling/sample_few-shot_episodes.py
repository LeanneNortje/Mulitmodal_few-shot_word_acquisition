#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2022
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

from pathlib import Path
import json
import re
import numpy as np
from tqdm import tqdm

K = 5
num_episodes = 1000

files = "../../Datasets/spokencoco/SpokenCOCO"
val_fn = Path(files) / 'SpokenCOCO_val.json'
val = {}

vocab = []
with open('../data/test_keywords.txt', 'r') as f:
    for keyword in f:
        vocab.append(' '.join(keyword.split()))

alignments = {}
prev = ''
prev_wav = ''
prev_start = 0
with open(Path('../../Datasets/spokencoco/SpokenCOCO/words.txt'), 'r') as f:
    for line in f:
        wav, start, stop, label = line.strip().split()
        if label in vocab or (label == 'hydrant' and prev == 'fire' and wav == prev_wav):
            if wav not in alignments: alignments[wav] = {}
            if label == 'hydrant' and prev == 'fire': 
                label = prev + " " + label
                start = prev_start
            if label not in alignments[wav]: alignments[wav][label] = (int(float(start)*50), int(float(stop)*50))
        prev = label
        prev_wav = wav
        prev_start = start

with open(val_fn, 'r') as f: data = json.load(f)
data = data['data']

for entry in data: 
    im = entry['image']
    for caption in entry['captions']:
        for word in vocab:
            if re.search(word, caption['text'].lower()) is not None and Path(caption['wav']).stem in alignments:
                if word not in val: val[word] = []
                val[word].append((im, caption['wav'], caption['speaker']))

test_episodes = {}

matching_set = {}

##################################
# Test matching set 
##################################

for entry in data:
    im = entry['image']
    if im not in matching_set: matching_set[im] = set()
    for caption in entry['captions']:
        
        for word in vocab:
            if re.search(word, caption['text'].lower()) is not None:
                if im not in matching_set: matching_set[im] = set()
                matching_set[im].add(word)
        # used_images.add(im)
test_episodes['matching_set'] = matching_set
print(len(matching_set))

##################################
# Test queries  
##################################

for word in vocab:

    instances = np.random.choice(np.arange(0, len(val[word])), num_episodes)        
    for episode_num in tqdm(range(num_episodes)):

        if episode_num not in test_episodes: test_episodes[episode_num] = {'queries': {}, 'matching_set': {}}
        entry = val[word][instances[episode_num]]
        test_episodes[episode_num]['queries'][word] = (entry[1], entry[2])
        test_episodes[episode_num]['matching_set'][word] = (entry[0])


for episode_n in range(num_episodes):
    if len(test_episodes[episode_num]['queries']) != 5 or len(test_episodes[episode_num]['matching_set']) != 5:
        print("BUG")
test_save_fn = '../data/test_episodes'
np.savez_compressed(
    Path(test_save_fn).absolute(), 
    episodes=test_episodes
    )