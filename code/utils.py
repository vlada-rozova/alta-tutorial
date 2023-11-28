import numpy as np
import pandas as pd
import re
from sklearn.model_selection import StratifiedGroupKFold

def get_filename(patient_id, report_no, file_format='ann'):
    """
    Return the filename of the annotation file.
    """
    return "pt" + str(patient_id) + "_r" + str(report_no) + "." + file_format



def get_cv_strategy(n_splits=10):
    """
    Return the CV object.
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=3)



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



def read_annotations(df, path):
    """
    Parse and post-process .ann files.
    """
    # Parse annotation files
    concepts, relations = parse_ann_files(df, path)
    
    # Separate discontinuous concepts
    concepts = handle_discont_concepts(concepts)
    
    # Save the extracted concepts and relations
    concepts.to_csv("../datasets/gold_concepts.csv", index=False)
    relations.to_csv("../datasets/gold_relations.csv", index=False)
    
    return concepts



def parse_ann_files(df, path):
    """
    Parse .ann files and write into a dataframe of gold standard concepts.
    """
    # Create dataframes to store annotations
    concepts = pd.DataFrame(columns=['histopathology_id', 'patient_id', 'report_no', 
                                     'concept_id', 'concept', 'phrase', 'position', 'start_char', 'end_char'])
    relations = pd.DataFrame(columns=['histopathology_id', 'patient_id', 'report_no', 
                                      'relation_id', 'relation', 'arg1', 'arg2'])

    for _, x in df.iterrows():
        # Define filename
        filename = get_filename(x.patient_id, x.report_no, file_format='ann')

        # Open and read annotation file
        with open(path + filename, 'r') as f:
            annotation = f.readlines()

        if annotation:    
            # Loop over each line of the annotation file
            for line in annotation:

                # Concept
                if re.match("T", line):

                    # Create an entry containing concept ID, category, position and the raw text
                    substrings = line.strip().split('\t')
                    concept_id = substrings[0]
                    concept = substrings[1].split(maxsplit=1)[0]
                    position = substrings[1].split(maxsplit=1)[1]
                    start_char, end_char = re.split(' |;', position)[-2:]
                    text = substrings[2]

                    tmp = pd.DataFrame({
                        'histopathology_id': x.histopathology_id,
                        'patient_id': x.patient_id, 
                        'report_no': x.report_no, 
                        'concept_id': concept_id, 
                        'concept': concept, 
                        'phrase': text,
                        'position': position, 
                        'start_char': int(start_char),
                        'end_char': int(end_char),
                    }, index=[0])

                    # Add to the table of concepts
                    concepts = pd.concat([concepts, tmp], axis=0, ignore_index=True)

                # Relation
                elif re.match("R", line):

                    # Create an entry containing relation ID, type and IDs of the arguments
                    substrings = line.strip().split()
                    relation_id = substrings[0]
                    relation = substrings[1]
                    arg1 = substrings[2].split(':')[1]
                    arg2 = substrings[3].split(':')[1]

                    tmp = pd.DataFrame({
                        'histopathology_id': x.histopathology_id,
                        'patient_id': x.patient_id, 
                        'report_no': x.report_no, 
                        'relation_id': relation_id, 
                        'relation': relation, 
                        'arg1': arg1, 
                        'arg2': arg2
                    }, index=[0])

                    # Add to the table of relations
                    relations = pd.concat([relations, tmp], axis=0, ignore_index=True)

    # Convert patient ID and report number to int
    concepts[['patient_id', 'report_no']] = concepts[['patient_id', 'report_no']].astype(int)
    relations[['patient_id', 'report_no']] = relations[['patient_id', 'report_no']].astype(int)
    
    return concepts, relations


def handle_discont_concepts(concepts):
    """
    Split discontinuos concepts.
    """
    # Discont concepts have ;-separated positions
    idx = concepts[concepts.position.str.contains(";")].index

    # Split discont concepts into a separate dataframe
    discont = concepts.iloc[idx].copy()
    concepts.drop(idx, inplace=True)
    
    # Loop over discont concepts extracting individual spans
    for _,x in discont.iterrows():
        spans = []
        i = 0
        for pos in x.position.split(';'):
            # Extract start and end char positions
            start_char, end_char = map(int, pos.split())
            # Calculate span length
            len_span = end_char - start_char
            # Extract span text
            phrase = x.phrase[i:i+len_span]
            # Add to list of spans
            spans.append((start_char, end_char, phrase))
            i = i + len_span + 1

        # Sort extracted spans by starting position
        spans = sorted(spans, key=lambda x: x[0])

        # Append extracted spans to the dataframe with gold standard concepts 
        for span in spans:
            tmp = x.copy()
            tmp['start_char'] = span[0]
            tmp['end_char'] = span[1]
            tmp['phrase'] = span[2]
            concepts = pd.concat([concepts, tmp.to_frame().T], axis=0, ignore_index=True)

    # Remove position column
    concepts.drop('position', axis=1, inplace=True)
        
    return concepts