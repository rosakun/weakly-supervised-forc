import pandas as pd
import numpy as np
import requests
from utils.utils import get_all_labels, find_superclass
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
import ast
import json

class FoRC4CLData:

    def __init__(self, 
                forc4cl_data_path):
        
        self.forc4cl_data_path = forc4cl_data_path
        self.forc4cl_df = pd.read_csv(self.forc4cl_data_path)
        self.length = self.forc4cl_df.shape[0]

        print("Got FoRC4CL data")

    # The following functions update a FoRC4CL dataset for missing values.

    def _get_doi_openalex(self):
        for index, row in self.forc4cl_df.iterrows():

            if pd.isna(row['doi']):
                search_url = "https://api.openalex.org/works?search=" + "\"" + row['title'] + "\""
                response = requests.get(search_url).json()
                #We need to handle cases where there are zero, one, or multiple responses.
                try:
                    if response['meta']['count'] == 1:
                        self.forc_df.at[index, 'doi'] = response['results'][0]['doi']
                    elif  response['meta']['count'] > 1:
                        for result in response['results']:
                            if result['title'] == row['title']:
                                self.forc4cl_df.at[index, 'doi'] = result['doi']
                                self.forc4cl_df['doi'] = self.forc4cl_df['doi'].str.replace('https://doi.org/', '', regex=False)
                                break                    
                except KeyError:
                    print("Error with response: ", response)
        print("Got DOIs from OpenAlex")

    def _save_updated_csv(self, output_path):
        self.forc_df.to_csv(output_path, index=False)
        print(f"Updated CSV saved to {output_path}")

    # The next functions are used for model training and labeling ACL datasets.

    def _get_documents(self,lowercase=False,stem=False, full_text=False): #This should concatenate title, abstract, publisher, booktitle, author, venue 
        # Preprocessing: Removing newlines from the text
        X = []
        columnslist = ['title', 'abstract', 'publisher', 'booktitle', 'author', 'venue']
        if full_text:
            columnslist.append('full_text')
        for index, row in self.forc4cl_df.iterrows():
            document = ""
            for column in columnslist:
                if not pd.isna(row[column]):
                    document += row[column].replace('\n', ' ') + " "
                    document = document[:-1]
            if lowercase:
                document = document.lower()
            if stem:
                stemmer = PorterStemmer()
                words = word_tokenize(document)
                stemmed = [stemmer.stem(word) for word in words]
                document = " ".join(stemmed)
            X.append(document[:-1])
        
        return X

    def _get_labels(self): #This should return numerical representations of the labels in Level1, Level2, Level3
        # Get the full list of labels in FoRC4CL and create a dictionary mapping each label to an index
        all_labels = get_all_labels()
        label_to_index = {label: idx for idx, label in enumerate(all_labels)}

        # Get a list of the labels for the documents
        y_train_labels = []
        for index, row in self.forc4cl_df.iterrows():
            row_labels = []
            for column in ['Level1', 'Level2', 'Level3']:
                if not pd.isna(row[column]):
                    cleaned_topics = row[column].replace('[', '').replace(']', '').replace("'", "").split(', ')
                    for topic in cleaned_topics:
                        row_labels.append(topic)
            y_train_labels.append(sorted(row_labels))
        
        
        n_samples = self.length
        n_labels = len(all_labels)

        # Initialize binary indicator matrix
        binary_matrix = np.zeros((n_samples, n_labels), dtype=int)

        for i, labels in enumerate(y_train_labels):
            for label in labels:
                binary_matrix[i, label_to_index[label]] = 1

        return binary_matrix
    

    def _write_predictions_to_new_file(self, predictions, outputfile):
        # Ensure columns can store mixed types
        self.forc4cl_df["Level1"] = self.forc4cl_df["Level1"].astype(object)
        self.forc4cl_df["Level2"] = self.forc4cl_df["Level2"].astype(object)
        self.forc4cl_df["Level3"] = self.forc4cl_df["Level3"].astype(object)
        for index, row in self.forc4cl_df.iterrows():
            level1, level2, level3 = predictions[index] if len(predictions[index]) == 3 else print("Hard failure")
            self.forc4cl_df.at[index, "Level1"] = str(predictions[index][0])
            if not predictions[index][1] == []:
                self.forc4cl_df.at[index, "Level2"] = str(predictions[index][1])
            if not predictions[index][2] == []:
                self.forc4cl_df.at[index, "Level3"] = str(predictions[index][2])
        self.forc4cl_df.to_csv(outputfile, index=False,encoding='utf-8')



    
    # The final functions are used to analyse the dataset.
            
    def _count_missingvalues(self):
        print("Total rows:", self.forc4cl_df.shape[0])
        for column in self.forc4cl_df.columns:
            null_count = self.forc4cl_df[column].isnull().sum().sum()
            print(column, null_count)

    def _count_missingclasses(self):
        all_labels = get_all_labels()
        for index, row in self.forc4cl_df.iterrows():
            if pd.notna(row['Level1']):
                level1 = ast.literal_eval(row['Level1'])
                for label in level1:
                    if label in all_labels:
                        all_labels.remove(label)
            if pd.notna(row['Level2']):
                level2 = ast.literal_eval(row['Level2'])
                for label in level2:
                    if label in all_labels:
                        all_labels.remove(label)
            if pd.notna(row['Level3']):
                level3 = ast.literal_eval(row['Level3'])
                for label in level1:
                    if label in all_labels:
                        all_labels.remove(label)
        print("Number of classes missing:", len(all_labels))
        print(all_labels)
    
    def _count_instances_per_class(self, by_level=False):
        print("Number of rows:", len(self.forc4cl_df))
        def process_level(column_name):
            level_dict = {}
            for index, row in self.forc4cl_df.iterrows():
                if not pd.isna(row[column_name]):
                    cleaned_topics = row[column_name].replace('[', '').replace(']', '').replace("'", "").split(', ')
                    for topic in cleaned_topics:
                        level_dict[topic] = level_dict.get(topic, 0) + 1
            # Sort the dictionary by values in descending order
            return dict(sorted(level_dict.items(), key=lambda item: item[1], reverse=True))

        # Process and sort each level
        sorted_level_1 = process_level('Level1')
        sorted_level_2 = process_level('Level2')
        sorted_level_3 = process_level('Level3')

        if by_level:
            sorted_level_1_dict = dict(sorted(sorted_level_1.items(), key=lambda item: item[1]))
            sorted_level_2_dict = dict(sorted(sorted_level_2.items(), key=lambda item: item[1]))
            sorted_level_3_dict = dict(sorted(sorted_level_3.items(), key=lambda item: item[1]))
            print("Level 1 values:")
            for key, value in sorted_level_1_dict.items():
                print(f"{key}: {value}")
            print("Level 2 values:")
            for key, value in sorted_level_2_dict.items():
                print(f"{key}: {value}")
            print("Level 3 values:")
            for key, value in sorted_level_3_dict.items():
                print(f"{key}: {value}")
        else:
            combined_dict = {**sorted_level_1, **sorted_level_2, **sorted_level_3}

            # Sort by values and keep the result as a dictionary
            sorted_dict = dict(sorted(combined_dict.items(), key=lambda item: item[1]))

            for key, value in sorted_dict.items():
                print(f"{key}: {value}")


    ## This part of the code creates datasets that can be used to train BERT

    def _convert_to_BERT_format(self):
        content_columns = ['acl_id', 'abstract', 'url', 'publisher', 'year', 'month', 'booktitle', 'author', 'title', 'doi', 'venue', 'data_index']
        bert_df = self.forc4cl_df[content_columns]
        
        labels = self._get_labels()
        class_labels = get_all_labels()
        
        # Assuming labels is a list of class labels
        new_columns = {label: [row[i] for row in labels] for i, label in enumerate(class_labels)}

        # Efficiently add all new columns at once
        bert_df = pd.concat([bert_df, pd.DataFrame(new_columns)], axis=1)

        bert_df.to_csv("forc4cl_BERT_test.csv")

    ## Makes a dataset containing one example per label

    def _one_instance_per_label(self):
        remaining_labels = set(get_all_labels())
        new_df = []
        for index, row in self.forc4cl_df.iterrows():
            labels = []
            if pd.notna(row['Level1']):
                level1 = ast.literal_eval(row['Level1'])
                labels += level1
            if pd.notna(row['Level2']):
                level2 = ast.literal_eval(row['Level2'])
                labels += level2
            if pd.notna(row['Level3']):
                level3 = ast.literal_eval(row['Level3'])
                labels += level3
            
            for label in labels:
                if label in remaining_labels:
                    new_df.append(row)
                    remaining_labels.difference_update(labels)  # Remove all labels in 'labels' from remaining_labels
                    break  # Ensure only one instance per label
                    
        pd.DataFrame(new_df).to_csv("sanitycheck.csv")

    # Gets rid of labels whose supercategories aren't predicted
    def _remove_labels_without_superclass(self):
        new_df = []

        with open("Taxonomy4CL_v1.0.1.json", "r", encoding="utf-8") as file:
            tree = json.load(file)

        for index, row in self.forc4cl_df.iterrows():
            Level2Flag = False

            if pd.notna(row['Level1']):
                level1 = ast.literal_eval(row['Level1'])
            else:
                level1 = []

            if pd.notna(row['Level2']):
                Level2Flag = True
                level2 = ast.literal_eval(row['Level2'])
                print("Old level 2:", level2)
            else:
                level2 = []

            if pd.notna(row['Level3']):
                level3 = ast.literal_eval(row['Level3'])
            else:
                level3 = []

            # Filter Level 3 labels (remove if their superclass is not in Level 2)
            level3 = [label for label in level3 if Level2Flag and find_superclass(tree, label) in level2]

            # Filter Level 2 labels (remove if their superclass is not in Level 1)
            level2 = [label for label in level2 if find_superclass(tree, label) in level1]

            # Create a new instance of the row with updated Level2 and Level3
            new_row = row.copy()
            if level2 == []:
                new_row['Level2'] = ""
            else:
                new_row['Level2'] = str(level2)  # Convert list back to string
            if level3 == []:
                new_row['Level3'] = ""
            else:
                new_row['Level3'] = str(level3)  # Convert list back to string

            new_df.append(new_row)

        # Convert new_df back into a DataFrame
        new_dataframe = pd.DataFrame(new_df)
        new_dataframe.to_csv("sanitycheck_fixed.csv")


            
            
            

        



if __name__ == '__main__':
    data = FoRC4CLData('sanitycheck.csv')
    data._remove_labels_without_superclass()

