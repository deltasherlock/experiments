# -*- coding: utf-8 -*-
"""
Created by Vladimir Pchelin on 10/28/17.

Copyright © 2017 Vladimir Pchelin. All rights reserved.
"""


import yaml
import os


def get_label_to_tokens(filename):
    """
    Returns the corpus: a dictionary from labels to sets of tokens.
    """
    label_to_tokens = dict()
    with open(filename) as f:
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
    Returns labels, not all, that have sets of tokens identical to other labels.
    From each group of identical labels one label goes to representatives. 
    All the other labels from each group go to <duplicates>.
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
                print('Duplicates: {0} = {1}'.format(label, other_label))
    return duplicates
  
       
def get_rules_per_label(label, label_to_tokens, token_to_labels,
                        label_to_token_groups, limit = 1, max_index = 5):
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
        if index > max_index:
            break
        for token in label_to_token_groups[label][index]:
            rule = []
            rule.append((token, 'unique to', str(index)))
            for other_label in token_to_labels[token]:
                if other_label == label:
                    continue
                plus_diff = label_to_tokens[label] - label_to_tokens[other_label]
                minus_diff = label_to_tokens[other_label] - label_to_tokens[label]
                assert (len(plus_diff) + len(minus_diff)) > 0
                plus_diff -= used_tokens
                minus_diff -= used_tokens
                if len(plus_diff) > 0:
                    rule.append((list(plus_diff)[0], 'inside vs', other_label))
                elif len(minus_diff) > 0:
                    rule.append((list(minus_diff)[0], 'outside vs', other_label))
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
              limit = 1, max_index = 5):
    """
    Generates a dictionary from labels to sets of rules.
    
    See description of <get_rules_per_label> for more details.
    """
    rules = dict()
    for label in label_to_token_groups:
        rules[label] = get_rules_per_label(label, label_to_tokens,
             token_to_labels, label_to_token_groups, limit, max_index)
    return rules


#
# Generate rules from the corpus
#
# Get label to tokens corpus from a file (apt or yum / paths or tuples or names)
label_to_tokens = get_label_to_tokens('~/Desktop/vladimir/apt/tuples')
# Filter out labels given by yum that refer to i686 architecture
label_to_tokens = {k: v for k, v in label_to_tokens.items() if k[-5:] != '.i686'} 
# Get the inverse map
token_to_labels = get_token_to_labels(label_to_tokens)
# Get the map from labels to categorized tokens
label_to_token_groups = get_label_to_token_groups(token_to_labels)
# Find duplicates
duplicates = get_duplicates(label_to_tokens, token_to_labels, label_to_token_groups)
# Filter out duplicates from the corpus
label_to_tokens = {k: v for k, v in label_to_tokens.items() if k not in duplicates} 
# Again get the inverse map
token_to_labels = get_token_to_labels(label_to_tokens)
# Again get the map from labels to categorized tokens
label_to_token_groups = get_label_to_token_groups(token_to_labels)
# Generate rules for all labels      
rules = get_rules(label_to_tokens, token_to_labels, label_to_token_groups, 1, 5)
# Free memory
del duplicates
del label_to_token_groups
del token_to_labels
del label_to_tokens


#
# Test on Anthony's data
#
def read_anthony_data(dirname):
    """
    Read in data provided by Anthony.

    By default only 'union' files are used!!!
    """
    anthony_data = dict()
    filenames = os.listdir(dirname)
    for filename in filenames:
        if 'union' not in filename: # By default only 'union' files are used
            continue
        with open(dirname + filename) as f:
            filedata = yaml.load(f)
        label = filedata['label']
        changes = filedata['changes']
        if label not in anthony_data:
            anthony_data[label] = dict()
        anthony_data[label][filename] = changes
    return anthony_data


def check_rules_on_anthony_data(rules, anthony_data):
    """
    If it doesn't print anything this means everything is good.
    """
    labels = anthony_data.keys()
    for label in labels:
        label_rules = rules[label]
        if len(label_rules) == 0:
            print 'There are no rules for label {0}'.format(label)
            continue
        for filename, changes in anthony_data[label].items():
            for rule in label_rules:
                for triplet in rule:
                    token = triplet[0]
                    inside = (triplet[1] != 'outside vs')
                    if inside == (len([v for v in changes if token in v]) == 0):
                        print('A rule broke on triplet: {0} {1} {2} filename: {3}'.
                              format(triplet[0], triplet[1], triplet[2], filename))
                        break
        for other_label in labels:
            if other_label == label:
                continue
            for filename, changes in anthony_data[other_label].items():
                for rule in label_rules:
                    flag = True
                    for triplet in rule:
                        token = triplet[0]
                        inside = (triplet[1] != 'outside vs')
                        if inside == (len([v for v in changes if token in v]) == 0):
                            flag = False
                            break
                    if flag:
                        print('Identified {0} as {1} where the rule has triplet: {2} {3} {4}'
                              .format(filename, label, rule[0][0], rule[0][1], rule[0][2]))


# Read Anthony's data
anthony_data = read_anthony_data('~/Desktop/anthony_yaml/training/')
# Filter out rules for labels that are not in Anthony's data
rules = {k: v for k, v in rules.items() if k in anthony_data.keys()}
# Check the rule on data. If nothing is printed it's good.
check_rules_on_anthony_data(rules, anthony_data)