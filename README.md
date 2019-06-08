# meta-learned-embeddings
Meta-Reinforcement Learning For Learning Domain Specific Embeddings While Utilizing Cross-Domain Knowledge

### NECESSARY INSTALLS:
  * Create conda environment
  * pip install tensorboardx
  * pip install pytorch-pretrained-bert

### HOW TO RUN MAML W/ BERT INITIALIZATION:
  * cd to meta-learned-embeddings directory
  * python bert_main.py
  * Results will be in maml_output/few_shot directory
  * To run tensorboard: tensorboard --logdir maml_output/few_shot/
  
### HOW TO RUN BERT W/O MAML:
  * cd to meta-learned-embeddings directory
  * python bert_data_preprocessing.py
  * Zero-Shot:
     * python pytorch-pretrained-BERT/examples/run_classifier_zero_shot.py --do_train --do_eval
     * To run tensorboard: tensorboard --logdir bert_zero_shot_output/tb
  * Few-Shot:
     * python pytorch-pretrained-BERT/examples/run_classifier_few_shot.py --do_train --do_eval
     * To run tensorboard: tensorboard --logdir bert_few_shot_output/tb
  * Standard:
     * python pytorch-pretrained-BERT/examples/run_classifier_standard.py --do_train --do_eval
     * To run tensorboard: tensorboard --logdir bert_standard_output/tb
  * Results will be in bert_zero_shot_output, bert_few_shot_output, and bert_standard_output folders


              
