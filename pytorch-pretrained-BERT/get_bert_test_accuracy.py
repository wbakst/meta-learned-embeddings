import argparse
import csv
import logging
import os
import random
import sys



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--predictions_dir",
                        default=None,
                        type=str,
                        required=True,
                        )
    parser.add_argument("--true_labels_dir",
                        default=None,
                        type=str,
                        required=True,
                        )

    true_labels_dir = args.true_labels_dir
    predictions_dir = args.predictions_dir

    if not os.path.exists(predictions_dir):
        raise Exception('{} doesn't exist'.format(predictions_dir))
    if not os.path.exists(true_labels_dir):
        raise Exception('{} doesn't exist'.format(true_labels_dir))

    # key: example_id --> value: label
    true_labels_dict = {}
    predictions_dict = {}

    




    if __name__ == "__main__":
        main()
