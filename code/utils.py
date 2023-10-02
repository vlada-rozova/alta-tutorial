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