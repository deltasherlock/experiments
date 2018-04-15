#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Vladimir Pchelin on 10/28/17.
Modified by Emre Ates after 04/05/18

Copyright © 2017 Vladimir Pchelin. All rights reserved.
"""

import os
import logging
from pathlib import Path
import random
from random import randint
# from random import sample
import pdb

import yaml
from tqdm import tqdm

DATA_DIR = Path('/projectnb/peaclab-mon/deltasherlock/repository/')
OUTPUT_DIR = Path('/projectnb/peaclab-mon/deltasherlock/results/')

def main():
    """ Generate rules from the corpus, tests on test set """
    logformat = '%(asctime)s %(levelname)-7s %(message)s'
    logging.basicConfig(
        format=logformat, level=logging.DEBUG,
        filename=OUTPUT_DIR / 'newest_logs.txt', filemode='w',
    )
    streamer = logging.StreamHandler()
    streamer.setFormatter(logging.Formatter(logformat))
    logging.getLogger().addHandler(streamer)

    anthony_corpus = read_anthony_data(DATA_DIR / 'training/',
                                       union=False, exclude_app='apache')
    anthony_data = read_anthony_data(DATA_DIR / 'testing/',
                                     exclude_app='apache')
    applications = anthony_corpus.keys()

    # counter = 0
    for training_set_size, _ in enumerate(applications):
        if training_set_size % 10 != 0:
            continue
        logging.info('Starting rule based method for %d application training '
                     'set', training_set_size)
        for _ in range(5):
            training_set = random.sample(applications, training_set_size)
            rules = generate_rules(
                {app: files for app, files in anthony_corpus.items()
                 if app in training_set}
            )

            # Filter out rules for labels that are not in Anthony's data
            rules = {k: v for k, v in rules.items()
                     if k in anthony_data.keys()}

            predictions = predict_rules_on_data(
                rules, anthony_data, threshold=0.5)
            print(predictions)
            # parameters['training_apps'] = training_set

            # counter += 1
            # parameters['training_set_name'] = \
            #     'training-set-varying-%d' % counter
            # save_results(
            #     res_matrix, parameters, r'/mnt/data/results/repository',
            #     filename=parameters['training_set_name'] + '_' + str(
            #         round(parameters['avg_num_rules'])) + '_' + str(
            #             parameters['threshold']))


def generate_rules(corpus):
    """Generates rules from the given corpus"""
    label_to_tokens = transform_anthony_intersection(corpus)
    # Filter out labels given by yum that refer to i686 architecture
    label_to_tokens = {k: v for k, v in label_to_tokens.items()
                       if k[-5:] != '.i686'}
    # Get the inverse map
    token_to_labels = get_token_to_labels(label_to_tokens)
    # Get the map from labels to categorized tokens
    label_to_token_groups = get_label_to_token_groups(token_to_labels)
    # Find duplicates
    duplicates = get_duplicates(label_to_tokens, token_to_labels,
                                label_to_token_groups)
    # Filter out duplicates from the corpus
    label_to_tokens = {k: v for k, v in label_to_tokens.items()
                       if k not in duplicates}
    # Again get the inverse map
    token_to_labels = get_token_to_labels(label_to_tokens)
    # Again get the map from labels to categorized tokens
    label_to_token_groups = get_label_to_token_groups(token_to_labels)
    # Generate rules for all labels
    rules = get_rules(label_to_tokens, token_to_labels,
                      label_to_token_groups, limit=1)
    logging.info('Finished rule generation')
    return rules


def get_label_to_tokens(filename):
    """
    Returns the corpus: a dictionary from labels to sets of tokens.
    """
    label_to_tokens = dict()
    with open(filename, encoding='utf8') as f:
        content = f.readlines()
    for line in content:
        line = line[:-1]
        if line[:4] == '==> ' and line[-4:] == ' <==':
            label = line[4:-4]
        else:
            if label not in label_to_tokens:
                label_to_tokens[label] = set()
            label_to_tokens[label].add(line)
    return label_to_tokens


def get_token_to_labels(label_to_tokens):
    """
    Returns the inverse map: a dictionary from tokens to sets of labels.
    """
    token_to_labels = dict()
    for label in label_to_tokens:
        for token in label_to_tokens[label]:
            if token not in token_to_labels:
                token_to_labels[token] = set()
            token_to_labels[token].add(label)
    return token_to_labels


def get_label_to_token_groups(token_to_labels):
    """
    Returns a categorized corpus. It's a dictionary from labels to groups
    of tokens. These groups are indexed with natural numbers. Index of a
    group shows in how many labels each token from this group is present.
    """
    label_to_token_groups = dict()
    for token in token_to_labels:
        for label in token_to_labels[token]:
            index = len(token_to_labels[token])
            if label not in label_to_token_groups:
                label_to_token_groups[label] = dict()
            if index not in label_to_token_groups[label]:
                label_to_token_groups[label][index] = set()
            label_to_token_groups[label][index].add(token)
    return label_to_token_groups


def get_duplicates(label_to_tokens, token_to_labels, label_to_token_groups):
    """
    Returns labels, not all, that have sets of tokens identical to other
    labels. From each group of identical labels one label goes to
    representatives. All the other labels from each group go to <duplicates>.
    """
    duplicates = set()
    for label in sorted(label_to_tokens.keys()):
        if label in duplicates:
            continue
        first_index = sorted(label_to_token_groups[label].keys())[0]
        first_token = list(label_to_token_groups[label][first_index])[0]
        potential_duplicates = token_to_labels[first_token]
        for other_label in sorted(list(potential_duplicates)):
            if (other_label <= label) or (other_label in duplicates):
                continue
            if label_to_tokens[label] == label_to_tokens[other_label]:
                duplicates.add(other_label)
                logging.info(
                    'Duplicates: {0} = {1}'.format(label, other_label))
    return duplicates


def get_rules_per_label(label, label_to_tokens, token_to_labels,
                        label_to_token_groups, limit=1, max_index=0):
    """
    Generates rules, at most <limit>, for a specified <label>.

    Each rule is a list of <index> triplets.

    Each rules includes exactly one triplet
    of the format:
        (*) (<token>, 'unique to', <index>)
    It means that <token> appears exactly in <index> different labels including
    <label>. All these labels, except <label>, are listed exactly once in other
    triplets as <other_label>. A triplet of this format always goes first.

    Other triplets have the formats:
        (1) (<token>, 'inside vs', <other_label>) or
        (2) (<token>, 'outside vs', <other_label>)
    It means that <token> distinguishes <label> from <other_label>. Format (1)
    means that <token> is in <label> but not in <other_label>. Format (2) means
    that <token> is in <other_label> but not in <label>.

    Rules are ordered in a list according to <index>. There could be less rules
    than <limit>. Across all rules each token can appear only once in a triplet
    of format (*) and only once in a triplet of format (1) or (2). This
    guarantees that changes to one token will affect at most two rules. It's
    also guaranteed that rules have the smallest possible <indeces> under the
    requirement given above.
    """
    assert (label in label_to_token_groups)
    rules = []
    used_tokens = set()
    for index in sorted(label_to_token_groups[label].keys()):
        if index > max_index and max_index > 0:
            break
        for token in label_to_token_groups[label][index]:
            if token in used_tokens:
                continue
            rule = []
            rule.append((token, 'unique to', str(index)))
            for other_label in token_to_labels[token]:
                if other_label == label:
                    continue
                plus_diff = label_to_tokens[label] - \
                    label_to_tokens[other_label]
                minus_diff = label_to_tokens[other_label] - \
                    label_to_tokens[label]
                assert (len(plus_diff) + len(minus_diff)) > 0
                plus_diff -= used_tokens
                minus_diff -= used_tokens
                if len(plus_diff) > 0:
                    rule.append(
                        (list(plus_diff)[0], 'inside vs', other_label))
                elif len(minus_diff) > 0:
                    rule.append(
                        (list(minus_diff)[0], 'outside vs', other_label))
                else:
                    break
            if len(rule) < index:
                continue
            rules.append(rule)
            for triplet in rule:
                used_tokens.add(triplet[0])
            if len(rules) >= limit:
                return rules
    return rules


def get_rules(label_to_tokens, token_to_labels, label_to_token_groups,
              limit=1, max_index=5):
    """
    Generates a dictionary from labels to sets of rules.

    See description of <get_rules_per_label> for more details.
    """
    rules = dict()
    for label in label_to_token_groups:
        rules[label] = get_rules_per_label(
            label, label_to_tokens, token_to_labels,
            label_to_token_groups, limit, max_index)
    return rules


def read_anthony_data(dirname, union=False, rate=1, threshold=1000,
                      exclude_app=''):
    """
    Read in data provided by Anthony.

    By default only 'union' files are used!!!

    Returns
    -------
    anthony_data : dict[str, dict[str, set[str]]]
        first dict contains a list of each label,
        second dict contains each yaml file parsed belonging to the label
        the set of strings contains the '$permissions $filename' strings
    """
    logging.info("Reading anthony_data from %s", dirname)
    counter = dict()
    anthony_data = dict()
    filenames = os.listdir(dirname)
    for filename in tqdm(filenames):
        filename_label = filename.split('.')[0]
        if filename_label in counter and counter[filename_label] > threshold:
            continue
        if randint(0, 999)/1000.0 > rate:
            continue
        if 'yaml' not in filename:
            continue
        # By default only 'union' files are used
        if union and 'union' not in filename:
            continue
        if (not union) and 'union' in filename:
            continue
        if exclude_app in filename:
            continue

        with open(os.path.join(dirname, filename), encoding='utf8') as f:
            filedata = yaml.load(f)
        if 'label' not in filedata:
            logging.warning(
                str(os.path.join(dirname, filename)) + " missed label !!!!")
            continue
        label = filedata['label']
        label = ''.join([l for l in label if l.isalpha()])
        changes = set(filedata['changes'])
        for c in changes:
            c = c[4:]
        if label not in anthony_data:
            anthony_data[label] = dict()
        anthony_data[label][filename] = changes
        if label not in counter:
            counter[label] = 0
        counter[label] += 1
    return anthony_data


def transform_anthony_intersection(data):
    res = dict()
    for label in data:
        for filename in data[label]:
            if label not in res:
                res[label] = dict()
            for token in data[label][filename]:
                if token not in res[label]:
                    res[label][token] = 1
                else:
                    res[label][token] += 1
            # res[label] = res[label].intersection(data[label][filename])
    newres = dict()
    for label in res:
        newres[label] = set()
        maxval = max(res[label].values())
        for token in sorted(res[label], key=res[label].get, reverse=True):
            if res[label][token] != maxval and len(newres[label]) > 50:
                break
            if res[label][token] < 0.94 * maxval and len(newres[label]) >= 40:
                break
            if res[label][token] < 0.88 * maxval and len(newres[label]) >= 26:
                break
            if res[label][token] < 0.8 * maxval and len(newres[label]) >= 16:
                break
            if res[label][token] < 0.7 * maxval and len(newres[label]) >= 10:
                break
            if res[label][token] < 0.6 * maxval and len(newres[label]) >= 8:
                break
            if res[label][token] < 0.5 * maxval and len(newres[label]) >= 6:
                break
            newres[label].add(token)
    return newres


def transform_anthony_data(data):
    res = dict()
    for k, v in data.items():
        kk = list(v.keys())[0]
        res[k] = set(v[kk])
    return res


def predict_rules_on_data(rules, dataset, threshold=1):
    """ Get predictions for the given data

    Returns
    -------
    predictions: dict[str, list[str]]
        A dict of file names -> list of predictions
    """
    predictions = {}
    for true_label in dataset:
        for filename, changes in dataset[true_label].items():
            predictions[filename] = []
            for label_tested, label_rules in rules.items():
                n_rules_satisfied = 0
                n_rules = len(label_rules)
                pdb.set_trace()
                for rule in label_rules:
                    rule_satisfied = True
                    for triplet in rule:
                        token = triplet[0]
                        inside = (triplet[1] != 'outside vs')
                        if inside == (len([v for v in changes
                                           if token == v[-len(token):]]) == 0):
                            rule_satisfied = False
                            break
                    if rule_satisfied:
                        n_rules_satisfied += 1
                if (n_rules_satisfied / n_rules) >= threshold:
                    predictions[filename].append(label_tested)
    logging.info('Finished rule checking')
    return predictions


def save_results(res_matrix, parameters, dirname, filename):
    # df = pd.DataFrame(res_matrix)
    # df.to_csv(os.path.join(dirname, filename + "_table.cvs"))
    with open(os.path.join(dirname, filename + "_table.yaml"),
              encoding='utf8', mode='w') as f:
        yaml.dump(res_matrix, f)
    with open(os.path.join(dirname, filename + "_parameters.yaml"),
              encoding='utf8', mode='w') as f:
        yaml.dump(parameters, f)


if __name__ == '__main__':
    main()
