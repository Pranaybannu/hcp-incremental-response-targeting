All HCP data and rep plans are synthetic. No actual rep plan was provided.
Rep plan assumes 80 HCPs per territory selected by calls, annual visits and brand volume.
Overlap is partly constructed by this assumption; it is not evidence of real rep behavior.
Eligible universe is all HCPs with >=3 annual visits. Original validation subset is their intersection with the existing validation split.
Validation rows were included in final refit; that subset is no longer held out.
The existing LR model was trained on the full train+validation population, not restricted to >=3 visits.
All-universe rankings include training rows and are illustrative scoring, not unbiased model evaluation.
Probability ranks descending; HCP ID breaks ties. It predicts prescribing, not incremental visit response.
Matching is by hcp_id. Synthetic rep plan is separate from randomized visit_assigned.
