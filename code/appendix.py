#!/usr/bin/env python3

from pathlib import Path

from rule_based_class import RuleBased


def main():
    changesets = {}
    for changeset in Path(
            '/home/ates/deltasherlock/centos-files').glob('*.changes'):
        with changeset.open() as f:
            changesets[changeset.stem] = [x.strip() for x in f]
        print(changesets)
        break


if __name__ == '__main__':
    main()
