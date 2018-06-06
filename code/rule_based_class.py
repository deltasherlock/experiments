#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging


class RuleBased:
    """ scikit-style wrapper """
    def __init__(self, threshold=0.5, max_index=20):
        self.threshold = threshold
        self.max_index = max_index

    def fit(self, X, y, csids=None):
        label_to_tokens = self._transform_anthony_intersection(X, y)
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
                          label_to_token_groups, limit=1,
                          max_index=self.max_index)

        # Filter out rules for labels that are not in Anthony's data
        self.rules = {k: v for k, v in rules.items() if k in y}
        logging.info('Finished rule generation')

    def predict(self, X, csids=None):
        predictions = []
        for changes in X:
            for label_tested, label_rules in self.rules.items():
                n_rules_satisfied = 0
                n_rules = len(label_rules)
                if n_rules == 0:
                    logging.info("%s has no rules", label_tested)
                    continue
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
                if (n_rules_satisfied / n_rules) >= self.threshold:
                    predictions.append(label_tested)
                    break
            else:  # No rule was satisfied
                predictions.append('???')
        logging.info('Finished rule checking')
        return predictions

    def _transform_anthony_intersection(self, changesets, labels):
        res = dict()
        for data, label in zip(changesets, labels):
            for token in data:
                if label not in res:
                    res[label] = dict()
                if token not in res[label]:
                    res[label][token] = 1
                else:
                    res[label][token] += 1
        newres = dict()
        for label in res:
            newres[label] = set()
            maxval = max(res[label].values())
            for token in sorted(res[label], key=res[label].get, reverse=True):
                mystery_vlad_condition = (
                    (res[label][token] != maxval
                        and len(newres[label]) > 50) or
                    (res[label][token] < 0.94 * maxval
                        and len(newres[label]) >= 40) or
                    (res[label][token] < 0.88 * maxval
                        and len(newres[label]) >= 26) or
                    (res[label][token] < 0.8 * maxval
                        and len(newres[label]) >= 16) or
                    (res[label][token] < 0.7 * maxval
                        and len(newres[label]) >= 10) or
                    (res[label][token] < 0.6 * maxval
                        and len(newres[label]) >= 8) or
                    (res[label][token] < 0.5 * maxval
                        and len(newres[label]) >= 6)
                )
                if mystery_vlad_condition:
                    break
                newres[label].add(token)
        return newres


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
    if label == 'subversion':
        import pdb
        pdb.set_trace()
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