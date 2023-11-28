import numpy as np
import pandas as pd
import re
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span, SpanGroup
from spacy.util import filter_spans
from utils import get_cv_strategy, get_feature_names

def clean_text(text, mode='clean text'):
    """
    Apply simple text preprocessing to reports and convert to lower case 
    or return the mapping for character positions.
    """
    # Create a list of character position indices
    mapping = list(range(0, len(text)))
    
    # Add a full stop before a section header
    pattern = re.compile("(\s*\n\n[A-Z]{5,})")
    
    # Adjust indices
    i = 0
    tmp = []
    for m in pattern.finditer(text):
        tmp += mapping[i:m.start()] + [np.nan]
        i = m.start()

    tmp += mapping[i:]
    mapping = tmp
        
    text = pattern.sub(r".\1", text)
    
    # Separate a plus sign from the preceding word with a space
    pattern = re.compile("(?<=\w)\+(?=\s)")
    
    # Adjust indices
    i = 0
    tmp = []
    for m in pattern.finditer(text):
        tmp += mapping[i:m.start()] + [np.nan]
        i = m.start()

    tmp += mapping[i:]
    mapping = tmp
    
    text = pattern.sub(r" +", text)

    # Separate a hyphen from the following word with a space
    pattern = re.compile("(?<=\s)-(?=\w)")
    
    # Adjust indices
    i = 0
    tmp = []
    for m in pattern.finditer(text):
        tmp += mapping[i:m.end()] + [np.nan]
        i = m.end()

    tmp += mapping[i:]
    mapping = tmp
    
    text = pattern.sub(r"- ", text)
                
    # Separate a question mark from the following word with a space
    pattern = re.compile("\?(?=\w)")
    
    # Adjust indices
    i = 0
    tmp = []
    for m in pattern.finditer(text):
        tmp += mapping[i:m.end()] + [np.nan]
        i = m.end()

    tmp += mapping[i:]
    mapping = tmp
        
    text = pattern.sub(r"? ", text)
        
    # Replace semicolon with a space
    pattern = re.compile(";")
    text = pattern.sub(r" ", text)
        
    # Remove multiple full stops
    pattern = re.compile("\.{2,}")
    
    # Adjust indices
    li = 0
    tmp = []
    for m in pattern.finditer(text):
        ri = m.start() + 1
        tmp += mapping[li:ri]
        li = m.end()
        
    tmp += mapping[li:]
    mapping = tmp

    text = pattern.sub(r".", text)
        
    # Remove multiple spaces
    pattern = re.compile("\s{2,}")
    
    # Adjust indices
    li = 0
    tmp = []
    for m in pattern.finditer(text):
        ri = m.start() + 1
        tmp += mapping[li:ri]
        li = m.end()
        
    tmp += mapping[li:]
    mapping = tmp
        
    text = pattern.sub(r" ", text)
    
    # Rstrip
    text = text.rstrip()
    
    # Convert all whitespace characters to space
    pattern = re.compile("\s")
    text = pattern.sub(r" ", text)
    
    # Convert to lowercase
    text = text.lower()
    
    if mode=='clean text':
        return text
    elif mode=='map positions':
        return mapping
    
    
    
def adjust_position(x):
    """
    Shift position indices by header length and adjust using the mapping.
    """    
    # Adjust position indices
    start_char = x.pos_mapping.index(x.start_char)
    end_char = x.pos_mapping.index(x.end_char-1) + 1
    
    return start_char, end_char



def preprocess_phrase(x):
    """
    Convert to lowercase and apply the same preprocessing as to report texts.
    """
    # Convert to lowercase
    x = x.lower()
    
    # Ensure the same preprocessing is applied to text and keywords
    pattern = re.compile("\s+")
    x = pattern.sub(r" ", x)
    
    pattern = re.compile("\?(?=\w)")
    x = pattern.sub(r"? ", x)
    
    return x


def create_vocab(report_ids, concepts, expand=False):
    """
    Collate a vocabulary of phrases annotated for each concept category.
    """
    feature_names = get_feature_names('concepts')
    
    # Create an empty dict to store vocabulary
    vocab = {ft: [] for ft in feature_names}
    
    # Update vocabulary with concept phrases
    vocab.update(concepts[concepts.histopathology_id.isin(report_ids) & 
                          concepts.concept.isin(feature_names)
                         ].groupby('concept').phrase.unique())
        
    # Preprocess and convert to a dict of sets
    vocab = {k: set(preprocess_phrase(t) for t in v) for k,v in vocab.items()}
            
    # Expand the Invasiveness category
    if expand:
        return expand_vocab(vocab)
    else:
        return vocab


    
def expand_vocab(vocab):
    """
    A custom function to expand the Invasiveness category with same-root words.
    """
    if any(['angio' in token for token in vocab['Invasiveness']]):
        vocab['Invasiveness'] = vocab['Invasiveness'].union(['angio-invasion',
                                                             'angio-invasive',
                                                             'angioinvasion',
                                                             'angioinvasive'])
        
    if any(['infiltrat' in token for token in vocab['Invasiveness']]):
        vocab['Invasiveness'] = vocab['Invasiveness'].union(['infiltrated',
                                                             'infiltrating',
                                                             'infiltration'])
            
    return vocab



def get_matcher(nlp, vocab):
    """
    Macth concepts and return raw matched spans.
    """
    # Initialise a matcher object
    matcher = PhraseMatcher(nlp.vocab)

    # Add patterns to matcher from vocabulary
    for ft in get_feature_names("concepts"):
        if ft in vocab:
            patterns = list(nlp.pipe(vocab[ft]))
            matcher.add(ft, None, *patterns)

    return matcher


def span_filter(x):
    """
    A custom function that filters spans to resolve overlapping. 
    """

    filtered_spans = SpanGroup(x.doc, name="filtered_spans", spans=[])

    j = 1
    for span in x.spans:
        if span in filtered_spans:
            j+=1
            continue
        try:
            if (span.start == x.spans[j].start) & (span.end == x.spans[j].end):
                filtered_spans.extend([s for s in [span, x.spans[j]] if s.label_!='ClinicalQuery'])
                j+=1
            else:
                filtered_spans.append(span)
                j+=1
        except:
            filtered_spans.append(span)
            
    x.doc.ents = filter_spans(filtered_spans)
                
    return x.doc



def detect_concepts(text, nlp, vocab):
    
     # Run NLP pipeline
    doc = text.apply(nlp)
    
    # Create matcher
    matcher = get_matcher(nlp, vocab)
    
    # Extract spans
    spans = doc.apply(lambda x: matcher(x, as_spans=True))

    # Custom span filter
    return pd.DataFrame({'doc':doc,'spans':spans}).apply(span_filter, axis=1)



def doc2concepts(df):
    # Create a dataframe to store annotations
    concepts = pd.DataFrame(columns=['histopathology_id', 'patient_id', 'report_no', 
                                     'concept', 'phrase',
                                     'start_char', 'end_char'])    

    for _, x in df.iterrows():        
        for ent in x.doc.ents:
                        
            tmp = pd.DataFrame({
                'histopathology_id': x.histopathology_id,
                'patient_id': x.patient_id, 
                'report_no': x.report_no, 
                'concept': ent.label_, 
                'phrase': x.clean_text[ent.start_char:ent.end_char],
                'start_char': ent.start_char,
                'end_char': ent.end_char,
                }, index=[0])
            
            # Add to the table of concepts
            concepts = pd.concat([concepts, tmp], axis=0, ignore_index=True) 
            
    return concepts