import pandas as pd
from si.data.dataset import Dataset

def read_csv(file_name: str, sep:str, features:bool, label:bool) -> pd.DataFrame:
    '''
Reads a CSV file and returns a pandas DataFrame as NumPy arrays for features and labels.

Parameters
----------
file_name: str
    The name of the CSV file to read.
sep: str
    The separator used in the CSV file. For example ',' for commas, ';' for semicolons.
features: bool
    Whether to return the column names of the features, by default False.
label: bool
    Whether to regard last column as label (output variable), by default False.

Returns
-------
x: np.ndarray
    The NumPy array of feature data (input variables). If `label` is True, it does not include the last column.
y: np.ndarray or
label_data : NumPy array
    The NumPy array of the label data, if `label=True`, otherwise None.
features: list or None
    A list of column names corresponding to features, if `features=True`, otherwise None
label: str or None
    Name of the column corresponding to the label if `label` is True, otherwise None.

'''
    
    df = pd.read_csv(file_name, sep=sep)


    if features and label:
        features = df.columns[:-1]
        label = df.columns[-1]
        x = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()

    elif features and not label:
        features = df.columns
        x = df.to_numpy()
        y = None

    elif not features and label:
        features = None
        labels= None
        x = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()

    else:
        x = df.to_numpy()
        y = None
        features = None
        label = None


    return x, y, features, label
        
def write_csv(file_name: str, dataset: Dataset, sep : str = ',', features: bool = False, label: bool = False) -> None:
    '''
    Writes a pandas DataFrame to a CSV file
    
    Parameters
    ----------
    file_name: str
        The name of the file to write to.
    dataset: pd.DataFrame
        The dataset object.
    sep: str
        The separator to use, by default ','.
    features: bool
        Whether the dataset has features, by default False.
    label: bool
        Wheter the dataset has a label, by default False.
        
    '''

    data = pd.dataframe(dataset.x)

    if features:
        data.columns = dataset.features

    if label:
        data[dataset.label] = dataset.y

    data.to_csv(file_name, sep=sep, index = False)