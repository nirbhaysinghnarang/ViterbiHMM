import csv
import os
import random
from validate import Validator


def create_submission(output_filepath, token_labels, token_inds):

    validator = Validator(token_labels=token_labels, token_indices=token_inds)
    label_dict = validator.format_output_labels()
    with open(output_filepath, mode='w') as csv_file:
        fieldnames = ['Id', 'Predicted']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for key in label_dict:
            p_string = " ".join([str(start)+"-"+str(end)
                                for start, end in label_dict[key]])
            writer.writerow({'Id': key, 'Predicted': p_string})
