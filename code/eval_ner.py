import numpy as np
import pandas as pd
import re


def get_feature_names(feature_set, tags=None):
    """
    Return the list of concepts, relations, or composite concepts.
    """
    if feature_set=='concepts':
        feature_names = ['ClinicalQuery', 'FungalDescriptor', 'Fungus', 'Invasiveness', 'Stain', 'SampleType', 
                'positive', 'equivocal', 'negative']
    elif feature_set=='relations':
        feature_names = ['positive-rel', 'equivocal-rel', 'negative-rel', 
                'fungal-description-rel', 'invasiveness-rel', 'fungus-stain-rel']
    elif feature_set=='composite':
        feature_names = ['affirmedFungalDescriptor', 'affirmedFungus', 'affirmedInvasiveness', 'affirmedStain',
                'negatedFungalDescriptor', 'negatedFungus', 'negatedInvasiveness', 'negatedStain']
    elif feature_set=='termsets':
        feature_names = ['preceding_positive', 'following_positive', 'preceding_negative', 'following_negative']
    
    if tags:
        return [tag + ft for ft in feature_names for tag in ["B-", "I-", "E-", "S-"]]
    else:
        return feature_names



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

    
    
def match_concepts(ids, expected, detected, feature_names):
    """
    For each report, calculate the number of TP, FP, and FN 
    based on the provided tables with gold standard and detected concepts.
    """
    # Create a dataframe to record the number of tp, fp, fn
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
        for c,v in candidates[candidates.concept.isin(feature_names)].concept.value_counts().items():
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



def evaluate_ner_cv(df, true_concepts, detected_concepts, feature_names):
    """
    Calculate the number of TP, FP, and FN per concept per validation split. 
    """
    # Count the number of TP, FP, and FN
    counts = match_concepts(df.histopathology_id, true_concepts, detected_concepts, feature_names)

    # Add fold numbers to calculate metrics over CV split
    counts = counts.join(df.set_index('histopathology_id')['val_fold'])

    # Presicion
    print(counts.groupby(['concept', 'val_fold']).apply(precision).groupby('concept').agg(['mean', 'std']).round(2))

    # Recall
    print(counts.groupby(['concept', 'val_fold']).apply(recall).groupby('concept').agg(['mean', 'std']).round(2))
        
        
        
def evaluate_ner(ids, true_concepts, detected_concepts, feature_names):
    """
    Calculate the number of TP, FP, and FN per concept.
    """
    # Count the number of TP, FP, and FN
    counts = match_concepts(ids, true_concepts, detected_concepts, feature_names)
    
    # Presicion
    print(counts.groupby('concept').apply(precision))
        
    # Recall
    print(counts.groupby('concept').apply(recall))
        