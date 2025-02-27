import pandas as pd
import os


class ACLData:

    # The ACL dataset is downloaded from Huggingface as a parquet file.
    def __init__(self, 
                acl_data_path = 'acl_anthology/acl-publication-info.74k.v2.parquet'):
        
        self.acl_data_path = acl_data_path
        self.acl_df = pd.read_parquet(acl_data_path)

        self.acl_df["Level1"] = None
        self.acl_df["Level2"] = None
        self.acl_df["Level3"] = None
        self.acl_df["venue"] = None
        self.acl_df["data_index"] = None


        self.acl_columns = self.acl_df.columns
        self.row_count = len(self.acl_df)

        print("Got ACL data")


    # Converts the file taken from the parquet to the same format as the FoRC4CL dataset. 
    # The resulting .csv file can be processed with the FoRC4CL class.
    def _convert_to_FoRC4CL_format(self):
        
        forc4cl_df = self.acl_df[["acl_id", "Level1", "Level2", "Level3", "abstract", "full_text", "url", "publisher", "year", "month", "booktitle", "author", "title", "doi", "venue", "data_index"]]
        forc4cl_df.to_csv("acl_with_fulltext.csv", index=False)


    # Adds the full text of the ACL dataset to the FoRC4CL dataset.
    def _add_fulltext_to_FoRC4CL(self, forcpath):
        forc_df = pd.read_csv(forcpath)
        
        forc_with_fulltext = forc_df.merge(self.acl_df[['acl_id', 'full_text']], on='acl_id', how='left')

        forc_with_fulltext.to_csv("forc4cl_fulltext/val_fulltext.csv", index=False, encoding='utf-8')

    # Counts the number of missing values in each column of the ACL dataset.
    def _count_missingvalues(self):
        print("Total rows:", self.row_count)
        for column in self.acl_columns:
            null_count = self.acl_df[column].isnull().sum().sum()
            print(column, null_count)






if __name__ == '__main__':
    acl = ACLData()
    acl._add_fulltext_to_FoRC4CL("forc4cl_doi/val_doi.csv")
        