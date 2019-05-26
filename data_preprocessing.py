import collections

########################################
############ DATA VARIABLES ############
########################################

DOMAINS = [
    'apparel',
    'automotive',
    'baby',
    'beauty',
    'books',
    'camera_photo',
    'cell_phones_service',
    'computer_video_games',
    'dvd',
    'electronics',
    'gourmet_food',
    'grocery',
    'health_personal_care',
    'jewelry_watches',
    'kitchen_housewares',
    'magazines',
    'music',
    'musical_instruments',
    'office_products',
    'outdoor_living',
    'software',
    'sports_outdoors',
    'tools_hardware',
    'toys_games',
    'video',
]
LABEL_TYPE = [
    't2',
    't4',
    't5'
]
SPLITS = [
    'train',
    'dev',
    'test'
]

########################################
############# DATA READING #############
########################################

vocabulary = set()
examples = collections.defaultdict(list)

for domain in DOMAINS:
    for label_type in LABEL_TYPE:
        for split in SPLITS:
            filename = './data/{}.{}.{}'.format(domain, label_type, split)
            key = './preprocessed_data/{}.{}.{}'.format(domain, label_type, split)
            with open(filename, 'r') as file:
                for line in file:
                    line = line.split()
                    examples[key].append(line)
                    for word in line[:-1]: 
                        if word.isalpha(): vocabulary.add(word)

########################################
############## ASSIGN IDS ##############
########################################

VOCAB_IDS = {'UNK' : len(vocabulary) + 1}
for i, word in enumerate(sorted(vocabulary)):
    VOCAB_IDS[word] = i+1
    
# Save Vocab IDS
filename = './preprocessed_data/vocab_ids'
with open(filename, 'w') as out:
    for word, wid in VOCAB_IDS.items():
        print('{} {}'.format(word, wid), file=out)

###########################################
# CONSTRUCT ID EXAMPLES AND WRITE TO FILE #
###########################################

new_examples = collections.defaultdict(list)
for filename, reviews in examples.items():
    with open(filename, 'w') as out:
        for ex in reviews:
            words, label = ex[:-1], ex[-1]
            new_ex = [VOCAB_IDS[word] if word in VOCAB_IDS else VOCAB_IDS['UNK'] for word in words] + [label]
            print(' '.join([str(i) for i in new_ex]), file=out)