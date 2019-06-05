import collections
import csv
import os

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
TEST_CATEGORIES = ['books','dvd','electronics','kitchen_housewares']
DEV_CATEGORIES = ['apparel','camera_photo','magazines','office_products']
TRAIN_CATEGORIES = []

for domain in DOMAINS:
    if domain not in TEST_CATEGORIES and domain not in DEV_CATEGORIES:
        TRAIN_CATEGORIES.append(domain)

vocabulary = set()
examples = collections.defaultdict(list)

bert_preprocessed_data_dir = './bert_preprocessed_data'
if not os.path.exists(bert_preprocessed_data_dir):
    os.makedirs(bert_preprocessed_data_dir)

for domain in DOMAINS:
    for label_type in LABEL_TYPE:
        for split in SPLITS:
            dir = './bert_preprocessed_data/{}/{}'.format(domain, label_type)
            if not os.path.exists(dir):
                os.makedirs(dir)

for domain in DOMAINS:
    for label_type in LABEL_TYPE:
        for split in SPLITS:
            filename = './data/{}.{}.{}'.format(domain, label_type, split)

            write_split = None
            if domain in TEST_CATEGORIES:
                write_split = 'test'
            elif domain in DEV_CATEGORIES:
                write_split = 'dev'
            elif domain in TRAIN_CATEGORIES:
                write_split = 'train'

            key = './bert_preprocessed_data/{}/{}/{}.tsv'.format(domain, label_type, write_split)
            with open(filename, 'r') as file:
                for line in file:
                    line = line.split()
                    examples[key].append(line)

###########################################
# CONSTRUCT ID EXAMPLES AND WRITE TO FILE #
###########################################
num_examples = 0

new_examples = collections.defaultdict(list)
for filename, reviews in examples.items():
    """
    if filename.endswith('test.tsv'):
        true_test_label_filename = filename[:filename.rfind(".")] + "_true_labels.tsv"
        true_test_label_fobj = open(true_test_label_filename, 'w', newline='\n')
        true_label_writer = csv.writer(true_test_label_fobj, delimiter='\t')
    """

    with open(filename, 'w', newline='\n') as out:
        writer = csv.writer(out, delimiter='\t')
        """
        if filename.endswith('test.tsv'):
            writer.writerow(['id', 'sentence'])
        """
        for ex in reviews:
            num_examples += 1
            #print("num_examples",num_examples)
            words, label = ex[:-1], ex[-1]
            if label == '-1':
                label = '0' # bert expects 0 or 1 label
            sentence = " ".join(words)
            ex_id = num_examples
            dummy_col_var = 'a'
            writer.writerow([ex_id, label, dummy_col_var, sentence])
            """
            if filename.endswith('train.tsv') or filename.endswith('dev.tsv'):
                #print([ex_id, label, dummy_col_var, words])

            elif filename.endswith('test.tsv'):
                writer.writerow([ex_id, sentence])
                true_label_writer.writerow([ex_id, label])

    if filename.endswith('test.tsv'):
        true_test_label_fobj.close()
    """

print("Preprocessed",num_examples, "for BERT")
