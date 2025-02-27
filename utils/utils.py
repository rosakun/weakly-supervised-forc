import numpy as np
import pandas as pd
import json
import glob
    
    
def get_all_labels(filename="Taxonomy4CL_v1.0.1.json",level='all'):
    #Returns a list of all labels in the dataset.
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

    #print("Total labels:", len(all_labels))
    return all_labels


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


def count_rows(file):
    df = pd.read_csv(file)
    print("Number of rows:", len(df))


def make_short_csv(file, output_file, num_rows):
    df = pd.read_csv(file)
    short_df = df.head(num_rows)
    short_df.to_csv(output_file, index=False)
    print(f"Saved {num_rows} rows to {output_file}")


def count_missing_values(files):
    for file in files:
        print("Now handling file:", file)
        df = pd.read_csv(file)
        total_missing = 0
        for column in df.columns:
            null_count = df[column].isnull().sum().sum()
            print(column, null_count)
            total_missing += null_count 
        print("Total missing values:", total_missing)


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



