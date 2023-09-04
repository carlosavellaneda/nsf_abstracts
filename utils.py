import glob
from typing import Tuple
import numpy as np
from tqdm import tqdm
import pandas as pd


def read_xml_data(directory_path: str) -> pd.DataFrame:
    """
    Read all xml files in a directory and return a pandas DataFrame

    Parameters
    ----------
    directory_path: str
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
        xml_data["file_name"] = xml_file.split("/")[-1]
        output_dataset.append(xml_data)

    output_dataset = pd.concat(output_dataset)
    return output_dataset.reset_index(drop=True)


def load_data(file_path: str) -> Tuple[np.array]:
    """
    Load data from a z-compressed numpy file

    Parameters
    ----------
    file_path: str
        Path to z-compressed numpy file

    Returns
    -------
    files: np.array
        Numpy array containing file names
    embeddings: np.array
        Numpy array containing embeddings
    """
    data = np.load(file_path, allow_pickle=True)
    files = data["file_names"]
    embeddings = data["embeddings"]
    return files, embeddings
