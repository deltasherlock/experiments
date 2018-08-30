#!/usr/bin/env python3

from pathlib import Path
from pprint import pprint

from rule_based_class import RuleBased


def main():
    labels = []
    changesets = []
    for changeset in Path(
            '/home/ates/deltasherlock/centos-files').glob('*.changes'):
        with changeset.open() as f:
            changesets.append([x.strip() for x in f])
        labels.append(changeset.stem)
    clf = RuleBased(string_rules=True)
    print(len(changesets), len(labels))
    clf.fit(changesets, labels)
    print(len(clf.rules))
    # for label, rules in clf.rules.items():
    #     pprint((label, rules), width=200, compact=True)


if __name__ == '__main__':
    main()
