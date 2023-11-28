import numpy as np
import pandas as pd


def assign_bioes_tags(concepts):
    """
    Assign BIOES tags to single and multiple token entities.
    """
    # Create dataframe to store concepts with BIOES tags
    concepts_bioes = pd.DataFrame(columns=concepts.columns)

    # Single-token entities
    concepts_bioes = concepts[concepts.doc.apply(len) == 1].copy()

    # Add the "S" tag
    concepts_bioes.concept = concepts_bioes.concept.apply(lambda x: "S-" + x)

    # Remove doc
    concepts_bioes.drop('doc', axis=1, inplace=True)
    
    # Multi-token entities
    for _,x in concepts[concepts.doc.apply(len) > 1].iterrows():

        # Loop over tokens
        for token in x.doc:

            # Skip if whitespace
            if token.is_space:
                continue

            # If the first token tag with "B-"
            if token.i==0:
                concept = "B-" + x.concept

            # If the last token tag with "E-"
            elif token.i+1==len(x.doc):
                concept = "E-" + x.concept

            # If in the middle tag with "I-"
            else:
                concept = "I-" + x.concept

            # Adjust start char position
            start_char = x.start_char + token.idx 

            tmp = pd.DataFrame({
                'histopathology_id': x.histopathology_id,
                'patient_id': x.patient_id, 
                'report_no': x.report_no, 
                'concept': concept, 
                'phrase': token,
                'start_char': start_char,
                'end_char': start_char + len(token),
            }, index=[0])

            # Add to the table of concepts
            concepts_bioes = pd.concat([concepts_bioes, tmp], axis=0, ignore_index=True) 

    # Sort BIOES tagged concepts
    concepts_bioes.sort_values(by=['histopathology_id', 'start_char'], inplace=True)
    
    print("After assigning BIOES tags there are a total of %d concepts." % concepts_bioes.shape[0])
    
    return concepts_bioes



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



def create_labels(x, concepts):
    """
    Assigns categories to gold standard concepts or 0, if a token was not annotated.
    """
    labels = []
    for token in x.doc:
        # Skip if whitespace
        if token.is_space:
            continue

        # Is there an annotated entity in the same location?
        concept = concepts.loc[(concepts.histopathology_id==x.histopathology_id) &
                               (concepts.start_char==token.idx), 'concept'] 
        
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
                ents['phrase'].append(x.report[start_char:end_char])
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
                ents['phrase'].append(x.report[start_char:end_char])
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