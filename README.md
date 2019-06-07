# meta-learned-embeddings
Meta-Reinforcement Learning For Learning Domain Specific Embeddings While Utilizing Cross-Domain Knowledge

### HOW TO RUN BERT W/O MAML:
  * cd to meta-learned-embeddings directory
  * run python bert_data_preprocessing.py
  * Zero-Shot:
     * run python pytorch-pretrained-BERT/examples/run_classifier_zero_shot.py --do_train --do_eval
  * Few-Shot:
     * run python pytorch-pretrained-BERT/examples/run_classifier_few_shot.py --do_train --do_eval
  * Standard:
     * run python pytorch-pretrained-BERT/examples/run_classifier_standard.py --do_train --do_eval
  * Results will be in bert_output folder


              
