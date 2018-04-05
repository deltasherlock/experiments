As agreed, I removed all changesets with apache2. Some terminology: "package" = "package label" = "label".

Every experiment has three files, such as:

1) apt-tuples_15_0.175_table

This is a matrix of results. Each changeset was tested against rules for each package label.
Rows are labels of changesets, "true labels". Columns are package labels we are trying to check. 
There is one extra column, called "_perfect_rule=num_files". This column just shows the number of
changesets tested for the corresponding row. An entry in this matrix from row R and column C shows
how many changesets with "true label" R are predicted to have package label C.

2) apt-tuples_15_0.175_logs

This is a file with logs. It contains records of all rules that broke. 

"A rule broke on triplet: amanda/chg-multi unique to 1 filename: amanda-server.28933.rp.ubx.ts.yaml"
means that tuple amanda/chg-multi is in the training changeset for amanda-server but not in the test
chageset with file name amanda-server.28933.rp.ubx.ts.yaml.

"Identified amanda-server.18122.rp.ubx.ts.yaml as samba where the rule has triplet:
644 /run/systemd/generator.late/smbd.service unique to 1" means that changeset
amanda-server.18122.rp.ubx.ts.yaml was identified as samba by the rule that says
"644 /run/systemd/generator.late/smbd.service" can belong only to samba package.

3) apt-tuples_15_0.175_parameters

This is a dictionary of parameters that are used. For example,

avg_num_files: the average number of changesets per label in Anthony's test data.
avg_num_rules: the average number of rules per label. In reality, I used either only one rule or
all available rules up to 20 different rules in my experiments.
threshold: we predict label X if "threshold" fraction of all rules predict label X, otherwise predict
absence of X.
training_set: what set was used for training. "Apt-tuples" means corpus generated with package manager apt.
Then only tuples, for example, "amanda/readme.txt", are used. "Anthony-intersect-training" means that for
each label I used intersection of all changesets from Anthony's training data as "true changesets" 
for training.