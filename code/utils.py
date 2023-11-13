import numpy as np
import pandas as pd
import re
### Helper functions to keep the main Jupiter notebook nice and clean.

def get_filename(patient_id, report_no, file_format='ann'):
    """
    Return the filename of the annotation file.
    """
    return "pt" + str(patient_id) + "_r" + str(report_no) + "." + file_format



def read_report(x, path):
    """
    Import report texts from .txt files.
    """
    # Define filename
    filename = get_filename(x.patient_id, x.report_no, file_format='txt')
    
    # Open and read text file
    with open(path + filename, 'r') as f:
        text = f.read()
    
    return text


### Evaluation
def match_concept_location(true_concept, candidates, mode='exact'):
    """
    For a given true concept, check if any of the candidate concepts 
    match exactly or overlap in their location.
    """
    if mode=='exact':
        return candidates[(candidates.start_char == true_concept.start_char) &  
                          (candidates.end_char == true_concept.end_char)].index
    if mode=='overlap':
        return candidates[
            (candidates.start_char <= true_concept.start_char) & (candidates.end_char >= true_concept.end_char) |
            (candidates.start_char >= true_concept.start_char) & (candidates.start_char <= true_concept.end_char) |
            (candidates.end_char >= true_concept.start_char) & (candidates.end_char <= true_concept.end_char)
        ].index

    
    
def match_concepts(ids, expected, detected):
    """
    For each report, calculate the number of TP, FP, and FN 
    based on the provided tables with gold standard and detected concepts.
    """
    # Create a dataframe to record the number of tp, fp, fn
    feature_names = utils.get_feature_names('concepts')
    idx = pd.MultiIndex.from_product([ids, 
                                      pd.CategoricalIndex(feature_names,
                                                          categories=feature_names,
                                                          ordered=True)
                                     ],
                                     names=['histopathology_id', 'concept'])
    col = ['tp', 'fp', 'fn']
    counts = pd.DataFrame(0, idx, col)

    # Calculate the number of tp, fp, fn
    for _id in ids:

        # Candidate concepts
        candidates = detected[detected.histopathology_id==_id].copy()

        # Loop over gold standard concepts 
        for _, true_concept in expected[expected.histopathology_id==_id].iterrows():

            # Check if there is a candidate concept in the same location
            matched_id = match_concept_location(true_concept, candidates, mode='overlap')

            # No concepts detected in this location: false negative
            if matched_id.empty:
                counts.loc[_id, true_concept.concept].fn += 1
            else:

                # The candidate matches on the concept category: true positive
                if (candidates.loc[matched_id, 'concept'] == true_concept.concept).any():
                    counts.loc[_id, true_concept.concept].tp += 1

                # The predicted cocept category is incorrect; false negative
                else:
                    counts.loc[_id, true_concept.concept].fn += 1

        # Calculate the number of false positives            
        for c,v in candidates.concept.value_counts().items():
            counts.loc[_id, c].fp = v - counts.loc[_id, c].tp


    return counts



def precision(x):
    try:
        return x.tp.sum() / (x.tp.sum()+x.fp.sum())
    except:
        return np.nan

def recall(x):
    try:
        return x.tp.sum() / (x.tp.sum()+x.fn.sum())
    except:
        return np.nan
        
def fscore(p, r):
    return 2 * (p * r) / (p + r)



def evaluate_ner():
    # Count the number of TP, FP, FN
    counts = match_concepts(df.histopathology_id, concepts, concepts_auto)
    # Add fold numbers to calculate metrics over CV split
    counts = counts.join(df.set_index('histopathology_id').fold)
    # CV presicion
    counts.groupby(['concept', 'fold']).apply(precision).groupby('concept').agg(['mean', 'std']).round(2)
    # CV recall
    counts.groupby(['concept', 'fold']).apply(recall).groupby('concept').agg(['mean', 'std']).round(2)