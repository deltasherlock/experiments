#!/usr/bin/env python3

from pathlib import Path

from rule_based_class import RuleBased


def main():
    for changeset in Path('/home/ates/deltasherlock/centos-files').glob('*.changes'):
        print(changeset)


if __name__ == '__main__':
    main()
