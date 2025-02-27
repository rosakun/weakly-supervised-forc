import numpy as np
import pandas as pd
import json
import glob


"""
Various functions that were useful for data preprocessing and analysis.
"""
    
# This function produces a list of all labels in the dataset.
# If you want to get only labels of a certain level, specify it with the 'level' parameter, e.g. level='lvl1'.
def get_all_labels(filename="Taxonomy4CL_v1.0.1.json",level='all'): 

    labels_lvl1 = []
    labels_lvl2 = []
    labels_lvl3 = []
    with open(filename, 'r') as f:
        data = json.load(f)
        for child in data['children']:
            labels_lvl1.append(child['name'].strip())
            try:
                for child2 in child['children']:
                    labels_lvl2.append(child2['name'].strip())

                    try:
                        for child3 in child2['children']:
                            labels_lvl3.append(child3['name'].strip())
                    except KeyError:
                        pass

            except KeyError:
                pass

    if level == 'lvl1':
        all_labels = sorted(labels_lvl1)
    elif level == 'lvl2':
        all_labels = sorted(labels_lvl2)
    elif level == 'lvl3':
        all_labels = sorted(labels_lvl3)
    elif level == 'all':
        all_labels = sorted(labels_lvl1 + labels_lvl2 + labels_lvl3)

    return all_labels



# This function concatenates all csv files in a folder into a single csv file.
def concat_csv_files(csvfolder, output_file):

    # Specify the folder containing your CSV files
    csv_folder = csvfolder + "/*.csv"

    # Get a list of all CSV files in the folder
    csv_files = glob.glob(csv_folder)

    # Read and concatenate all the CSV files
    combined_df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)

    print("Successfully combined the following files:")
    print(csv_files)



# Counts the number of rows in a .csv file.
def count_rows(file):
    df = pd.read_csv(file)
    print("Number of rows:", len(df))


# Save the top num_rows rows of a csv file to a new csv file.
def make_short_csv(file, output_file, num_rows):
    df = pd.read_csv(file)
    short_df = df.head(num_rows)
    short_df.to_csv(output_file, index=False)
    print(f"Saved {num_rows} rows to {output_file}")



# Given a class from the Taxonomy4CL taxonomy, this function returns the superclass of that class.
def find_superclass(tree, target_name):
    """
    Recursively searches for the target name in the tree and returns its parent category.

    :param tree: Dictionary representing the hierarchical structure
    :param target_name: The name to search for
    :param parent: The parent category (default: None)
    :return: The parent category (superclass) or None if not found
    """
    if "children" in tree:  # If the node has children
        for child in tree["children"]:
            if child["name"] == target_name:
                return tree["name"]  # Return the current node's name as superclass
            found = find_superclass(child, target_name)  # Recursively search
            if found:
                return found  # If found in subtree, return superclass
    return None  # If not found



