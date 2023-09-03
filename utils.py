import glob
from tqdm import tqdm
import pandas as pd


def read_xml_data(directory_path: str) -> pd.DataFrame:
    """
    Read all xml files in a directory and return a pandas DataFrame

    Parameters
    ----------
    directory_path : str
        Path to directory containing xml files

    Returns
    -------
    output_dataset: pd.DataFrame
        Pandas DataFrame containing all xml files in directory
    """
    all_xml_files = glob.glob(directory_path + "/*.xml")
    output_dataset = []
    for xml_file in tqdm(all_xml_files):
        xml_data = pd.read_xml(xml_file)
        output_dataset.append(xml_data)

    output_dataset = pd.concat(output_dataset)
    return output_dataset.reset_index(drop=True)


def remove_xml_breaks(dataset: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Remove xml breaks from a column in a pandas DataFrame

    Parameters
    ----------
    dataset : pd.DataFrame
        Pandas DataFrame containing xml data
    column_name : str
        Name of column containing xml data

    Returns
    -------
    dataset: pd.DataFrame
        Pandas DataFrame with xml breaks removed
    """
    dataset[column_name] = dataset[column_name].str.replace("&lt;br/&gt;", " ")
    dataset[column_name] = dataset[column_name].str.replace(" +", " ", regex=True)
    return dataset
