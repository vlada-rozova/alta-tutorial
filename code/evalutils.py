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



def compare_char_position(tc, candidates, mode='exact'):
    """
    For a given true concept, check if any of the candidate concepts 
    match exactly or overlap in their location.
    """
    if mode=='exact':
        cond1 = candidates.start_char == tc.start_char
        cond2 = candidates.end_char == tc.end_char
        return not candidates[cond1 & cond2].empty
    
    if mode=='overlap':
        cond1a = candidates.start_char <= tc.start_char
        cond1b = candidates.start_char >= tc.start_char
        cond2a = candidates.end_char >= tc.end_char
        cond2b = candidates.end_char <= tc.end_char
        cond3 = candidates.start_char <= tc.end_char
        cond4 = candidates.end_char >= tc.start_char
        
        return candidates[(cond1a & cond2a) | (cond1b & cond3) | (cond2b & cond4)].empty
   

    
def match_concepts(ids, true_concepts, detected_concepts, feature_names):
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

    for _,tc in true_concepts[true_concepts.histopathology_id.isin(ids) & 
                              true_concepts.concept.isin(feature_names)
                             ].iterrows():

        candidates = detected_concepts[(detected_concepts.histopathology_id==tc.histopathology_id) & 
                                       (detected_concepts.concept==tc.concept)]

        if compare_char_position(tc, candidates, mode='overlap'):
            counts.loc[tc.histopathology_id, tc.concept].fn += 1
        else:
            counts.loc[tc.histopathology_id, tc.concept].tp += 1

    for _,dc in detected_concepts[detected_concepts.histopathology_id.isin(ids) & 
                                  detected_concepts.concept.isin(feature_names)
                                 ].iterrows():

        candidates = true_concepts[(true_concepts.histopathology_id==dc.histopathology_id) & 
                                   (true_concepts.concept==dc.concept)]

        if compare_char_position(dc, candidates, mode='overlap'):
            counts.loc[dc.histopathology_id, dc.concept].fp += 1
            
    return counts



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
    
    

def evaluate_ner_cv(df, true_concepts, detected_concepts, feature_names):
    """
    Calculate the number of TP, FP, and FN per concept.
    """
    # Count the number of TP, FP, and FN
    counts = match_concepts(df.histopathology_id, true_concepts, detected_concepts, feature_names)
    
    # Add fold numbers to calculate metrics over CV split
    counts = counts.join(df.set_index('histopathology_id')['val_fold'])
    
    # Presicion
    print(counts.groupby(['concept', 'val_fold']).apply(precision).groupby('concept').agg(['mean', 'std']).round(2))

    # Recall
    print(counts.groupby(['concept', 'val_fold']).apply(recall).groupby('concept').agg(['mean', 'std']).round(2))