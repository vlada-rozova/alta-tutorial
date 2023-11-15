import numpy as np
import pandas as pd

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