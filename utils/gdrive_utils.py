import pandas as pd

def load_csv_from_gdrive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    df = pd.read_csv(url)
    return df
