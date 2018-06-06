# experiments
The scripts and stuff related to experiments for deltasherlock publications

# Changes going from `rule_based.py` to `rule_based_class.py`
* In `transform_anthony_intersection`, the mystery constants are grouped
  together, the input changed from `{label: {input_file: [tokens]}}` to
  scikit-style `X` (changesets), `y` (labels) input.
* `predict_rules_on_data` became the `RuleBased.predict` method. Also,
  it now predicts `???` for packages where no rule was satisfied. There was no
  previous check for this. Also, the predictions were `{label: [predictions]}`,
  now they are a list of predictions.
* In `get_rules`, `max_index` was used to skip certain files if they were common
  in too many applications. This caused some applications to have no rules, this
  limit is higher now.
