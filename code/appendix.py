#!/usr/bin/env python3

from pathlib import Path
from pprint import pprint

from rule_based_class import RuleBased


def main():
    labels = []
    changesets = []
    counter = 20
    for changeset in Path(
            '/home/ates/deltasherlock/centos-files').glob('*.changes'):
        with changeset.open() as f:
            changesets.append([x.strip() for x in f])
        labels.append(changeset.stem)
        if counter < 0:
            break
        else:
            counter -= 1
    clf = RuleBased(string_rules=True)
    clf.fit(changesets, labels)
    pprint(clf.rules)


if __name__ == '__main__':
    main()
