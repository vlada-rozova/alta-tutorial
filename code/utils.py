import numpy as np
import pandas as pd
import re

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
        with open(path + "annotations/" + filename, 'r') as f:
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

#     print("Extracted %d concepts and %d relations." % (concepts.shape[0], relations.shape[0]))
    
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
    
#     print("After handling discontinous concepts there are a total of %d concepts." % concepts.shape[0])
    
    return concepts


def get_cue_order(concepts, relations, cues_to_check):
    """
    Check if a cue is preceeding or following. 
    """
    def assign_termset(x):
        arg2_ids = relations[(relations.histopathology_id==x.histopathology_id) & 
                             (relations.arg1==x.concept_id)
                            ].arg2
        arg2_start_char = concepts[(concepts.histopathology_id==x.histopathology_id) & 
                                    concepts.concept_id.isin(arg2_ids)
                                   ].start_char
        return (x.start_char < arg2_start_char).any(), (x.start_char > arg2_start_char).any()

    # Only check order for positive and negative cue
    cues = concepts[concepts.concept.isin(cues_to_check)]
    
    # Determine if a cue is preceding and/or following
    return pd.DataFrame(cues.apply(assign_termset, axis=1).tolist(), 
                                                        index=cues.index)


def add_composite_concepts(concepts, relations, relations_to_add):
    """
    Combine concepts and relations into composite concepts.
    """
    def get_composite_concept(x, name):
        # Define the next vacant concept ID
        next_id = concepts[concepts.histopathology_id==x.histopathology_id].concept_id.apply(lambda x: 
                                                                                             int(x[1:])
                                                                                            ).max() + 1
        # Determine the object (Arg2) of a relation
        y = concepts[(concepts.histopathology_id==x.histopathology_id) & 
                     (concepts.concept_id==x.arg2)].iloc[0]
            
        # Create an entry containing concept ID, composite category, position and the raw text
        return pd.DataFrame({'histopathology_id': x.histopathology_id,
                             'patient_id': x.patient_id,
                             'report_no': x.report_no, 
                             'concept_id': 'T' + str(next_id), 
                             'concept': name + y.concept,
                             'phrase': y.phrase,
                             'start_char': y.start_char,
                             'end_char': y.end_char,
                            }, index=[0])
            
    for k,v in relations_to_add.items():
        # Loop over the dataframe with extracted relations
        for _, x in relations[relations.relation==k].iterrows():
            # Add to the table of concepts
            concepts = pd.concat([concepts, get_composite_concept(x, v)], axis=0, ignore_index=True)
    
     # Drop duplicated composite concepts
    concepts.drop_duplicates(subset=['histopathology_id', 'concept', 'start_char'], inplace=True, ignore_index=True)

#     print("Totalling %d concepts and composite concepts." % concepts.shape[0])
        
    return concepts



def read_annotations(df, path):
    # Parse annotation files
    concepts, relations = parse_ann_files(df, path)
    
    # Separate discontinuous concepts
    concepts = handle_discont_concepts(concepts)
    
    # Preceding and following termsets
    concepts[['preceding', 'following']] = get_cue_order(concepts, relations, 
                                                         ['positive', 'negative'])
    
    # Save the extracted concepts and relations
    concepts.to_csv("../datasets/gold_concepts.csv", index=False)
    relations.to_csv("../datasets/gold_relations.csv", index=False)
    return concepts
    
    # Create composite concepts
    concepts = add_composite_concepts(concepts, relations, 
                                      {'positive-rel': 'affirmed', 'negative-rel': 'negated'})
    
    # Save the extracted annotations
    concepts.to_csv("../datasets/gold_composite.csv", index=False)


def get_neighbors(doc, i):
    """
    Extract the previous and following tokens ignoring whitespaces.
    """
    if (i==0) or (i==1 and doc[i-1].is_space):
        prev_token = ""
    else:
        prev_token = doc[i-2].text if doc[i-1].is_space else doc[i-1].text
        
    if (i==len(doc)-1) or (i==len(doc)-2 and doc[-1].is_space):
        next_token = ""
    else:
        next_token = doc[i+2].text if doc[i+1].is_space else doc[i+1].text
    
    return prev_token, next_token
    
    
def create_features(doc):
    """
    Parses a doc and creates a dictionary of features for each token that is not a whitespace.
    """
    features = []
    
    for token in doc:
        
        # Skip if whitespace
        if token.is_space:
            continue
            
        # Get previous and next token
        prev_token, next_token = get_neighbors(doc, token.i)

        # Create a dict of features
        token_features = {
                'phrase': token.text,
                'start_char': token.idx,
                'end_char': token.idx + len(token),
                'is_capitilized': token.is_alpha and (token.text[0] == token.text.upper()[0]),
                'is_upper': token.is_upper,
                'is_lower': token.is_lower,
                'prefix1': token.text[:1],
                'prefix2': token.text[:2] if len(token)>1 else "",
                'prefix3': token.text[:3] if len(token)>2 else "",
                'suffix1': token.text[-1:],
                'suffix2': token.text[-2:] if len(token)>1 else "",
                'suffix3': token.text[-3:] if len(token)>2 else "",
                'prev_token': prev_token,
                'next_token': next_token,
                'has_hyphen': '-' in token.text,
                'is_alpha': token.is_alpha,
                'is_digit': token.is_digit,
                'is_sent_start': token.is_sent_start,
                'is_sent_end': token.is_sent_end,
                'is_punct': token.is_punct,
            }
        
        features.append(token_features)
        
    return features



def create_labels(x, true_concepts):
    """
    Assigns categories to gold standard concepts or 0, if a token was not annotated.
    """
    labels = []
    for token in x.doc:
        # Skip if whitespace
        if token.is_space:
            continue

        # Is there an annotated entity in the same location?
        concept = true_concepts.loc[(true_concepts.histopathology_id==x.histopathology_id) &
                                    (true_concepts.start_char==token.idx), 'concept'] 
        
        # Assign labels
        if concept.empty:
            labels.append("O")
        else:
            labels.append(concept.iloc[0])
    
    return labels



def prediction2concept_bioes(df):
    """
    Create a dataframe combining predicted concept category with character position.
    """
    concepts = pd.DataFrame(columns=['histopathology_id', 'patient_id', 'report_no',
                                     'concept', 'phrase', 'start_char', 'end_char'])

    for _,x in df.iterrows(): 

        if all(xx=='O' for xx in x):
            continue

        # Convert to dataframe
        tmp = pd.concat([pd.DataFrame(x.token_features, columns=['phrase', 'start_char', 'end_char']), 
                         pd.Series(x.y_pred, name='concept')],
                        axis=1)
        tmp = tmp[tmp.concept!='O']

        # Add metadata
        tmp['histopathology_id'] = x.histopathology_id
        tmp['patient_id'] = x.patient_id
        tmp['report_no'] = x.report_no    

        # Add to the table of detected concepts
        concepts = pd.concat([concepts, tmp], axis=0, ignore_index=True)   
        
    return concepts



def prediction2concept(df):
    """
    Create a dataframe combining predicted concept category with character position. 
    Combine BIE tokens into a single entity.
    """
    concepts = pd.DataFrame(columns=['histopathology_id', 'patient_id', 'report_no',
                                              'concept', 'phrase', 'start_char', 'end_char'])

    for _,x in df.iterrows(): 

        if all(xx=='O' for xx in x.y_pred):
            continue

        ents = {k:[] for k in ('concept', 'phrase', 'start_char', 'end_char')}

        for i,y in enumerate(x.y_pred):
            if y=="O":
                continue
            if y.startswith("S-"):
                # Record start and end char positions
                start_char = x.token_features[i]['start_char']
                end_char = x.token_features[i]['end_char']

                # Add single-token entity
                ents['concept'].append(y[2:])
                ents['phrase'].append(x.order_results[start_char:end_char])
                ents['start_char'].append(start_char)
                ents['end_char'].append(end_char)

                # Reset start_char, end_char (optional)
                start_char, end_char = None, None

            elif y.startswith("B-"):
                # Only track a multi-token entity if B is followed by I or E
                if x.y_pred[i+1].startswith("I-") or x.y_pred[i+1].startswith("E-"):
                    # Record start char position
                    start_char = x.token_features[i]['start_char']
                else:
                    continue

            elif y.startswith("I-"):
                if start_char:
                    continue
                else:
                    start_char = x.token_features[i]['start_char']

            elif y.startswith("E-"):
                # Record end char position
                end_char = x.token_features[i]['end_char']

                # Add multi-token entity
                ents['concept'].append(y[2:])
                ents['phrase'].append(x.order_results[start_char:end_char])
                ents['start_char'].append(start_char)
                ents['end_char'].append(end_char)

                # Reset start_char, end_char (optional)
                start_char, end_char = None, None

        # Convert to dataframe    
        tmp = pd.DataFrame(ents)

        # Add metadata
        tmp['histopathology_id'] = x.histopathology_id
        tmp['patient_id'] = x.patient_id
        tmp['report_no'] = x.report_no

        # Add to the table of detected concepts
        concepts = pd.concat([concepts, tmp], axis=0, ignore_index=True)   
        
    return concepts


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