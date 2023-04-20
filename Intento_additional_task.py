import pandas as pd
import numpy as np
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import xml.etree.ElementTree as ET


class DatasetCleaner:
    """
    A class for cleaning and filtering bilingual datasets stored in TMX files.
    """

    def __init__(self, file_names):
        """
        Initialize the DatasetCleaner object with the given TMX file names.
        """
        self.file_names = file_names
        self.df = self.read_tmx_files()

    def read_tmx_files(self):
        """
        Read the TMX files and return a Pandas DataFrame.
        """
        segments = []
        for file_name in self.file_names:
            tree = ET.parse(file_name)
            root = tree.getroot()

            for tu in root.iter('tu'):
                source_text = ""
                target_text = ""

                for tuv in tu.iter('tuv'):
                    lang = tuv.attrib['xml:lang']
                    seg = tuv.find('seg').text

                    if lang == 'en':
                        source_text = seg
                    elif lang == 'pt':
                        target_text = seg

                segments.append((source_text, target_text))

        df = pd.DataFrame(segments, columns=['source', 'target'])
        df['inner_id'] = range(1, len(df) + 1)
        return df

    def delete_duplicates(self):
        """
        Delete duplicates in source, target, and source+target.
        """
        self.df.drop_duplicates(subset=['source'], inplace=True)
        self.df.drop_duplicates(subset=['target'], inplace=True)
        self.df.drop_duplicates(subset=['source', 'target'], inplace=True)

    def token_count(self, text, lang):
        """
        Return the number of tokens in the given text using NLTK.
        """
        return len(word_tokenize(text, language=lang))

    def symbol_count(self, text):
        """
        Return the number of symbols in the given text.
        """
        return len(text)

    def sentence_count(self, text, lang):
        """
        Return the number of sentences in the given text using NLTK.
        """
        return len(sent_tokenize(text, language=lang))

    def has_link(self, text):
        """
        Check if the text contains a link using a regular expression.
        """
        return bool(
            re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))

    def clean_dataset(self):
        """
        Clean the dataset according to the given requirements.
        """
        self.delete_duplicates()
        self.apply_filters()

        self.df_filtered = self.df[
            (self.df['source_token_count'] >= 4) & (self.df['source_token_count'] <= 200) &
            (self.df['target_token_count'] >= 4) & (self.df['target_token_count'] <= 200) &
            (self.df['source_symbol_count'] >= 15) & (self.df['source_symbol_count'] <= 450) &
            (self.df['target_symbol_count'] >= 15) & (self.df['target_symbol_count'] <= 450) &
            (self.df['source_sentence_count'] == 1) & (self.df['target_sentence_count'] == 1) &
            (~self.df['has_link']) &
            (~self.df['digit_mismatch']) &
            (~self.df['invalid_target_start']) &
            (~self.df['invalid_source_middle'])
            ]

    def apply_filters(self):
        """
        Apply various filters to the dataset.
        """

        self.df['source_token_count'] = self.df['source'].apply(self.token_count, lang='english')
        self.df['target_token_count'] = self.df['target'].apply(self.token_count, lang='portuguese')
        self.df['source_symbol_count'] = self.df['source'].apply(self.symbol_count)
        self.df['target_symbol_count'] = self.df['target'].apply(self.symbol_count)
        self.df['source_sentence_count'] = self.df['source'].apply(self.sentence_count, lang='english')
        self.df['target_sentence_count'] = self.df['target'].apply(self.sentence_count, lang='portuguese')
        self.df['has_link'] = self.df['source'].apply(self.has_link) | self.df['target'].apply(self.has_link)
        self.df['digit_mismatch'] = self.df.apply(lambda row: set(re.findall(r'\d+', row['source'])) != set(re.findall(r'\d+', row['target'])), axis=1)
        self.df['invalid_target_start'] = self.df['target'].str.startswith('Então')
        self.df['invalid_source_middle'] = self.df['source'].apply(lambda x: 'actually' in x.split()[1:-1])

    def summarize(self):
        """
        Summarize the dataset by showing min and max inner_id, mean and median of source segment length in tokens and symbols.
        """
        min_inner_id = self.df_filtered['inner_id'].min()
        max_inner_id = self.df_filtered['inner_id'].max()
        mean_source_token_count = self.df_filtered['source_token_count'].mean()
        median_source_token_count = self.df_filtered['source_token_count'].median()
        mean_source_symbol_count = self.df_filtered['source_symbol_count'].mean()
        median_source_symbol_count = self.df_filtered['source_symbol_count'].median()

        summary = {
            'min_inner_id': min_inner_id,
            'max_inner_id': max_inner_id,
            'mean_source_token_count': mean_source_token_count,
            'median_source_token_count': median_source_token_count,
            'mean_source_symbol_count': mean_source_symbol_count,
            'median_source_symbol_count': median_source_symbol_count
        }

        return summary

    def save_to_excel(self, file_name):
        """
        Save the DataFrame to an Excel file.
        """
        self.df_filtered.to_excel(file_name, index=False)
"""
if __name__ == '__main__':
    file_names = ['hr-it.tmx', 'es-is.tmx', 'en-pl.tmx']
    cleaned_file_names = ['Cleaned_hr-it_dataset.csv', 'Cleaned_es-is_dataset.csv', 'Cleaned_en-pl_dataset.csv']

    for i, file_name in enumerate(file_names):
        dataset_cleaner = DatasetCleaner([file_name])
        dataset_cleaner.clean_dataset()
        dataset_cleaner.df_filtered.sort_values(by=['source_symbol_count'], ascending=True, inplace=True)
        summary = dataset_cleaner.summarize()
        print(f"Summary for {file_name}: {summary}")
        dataset_cleaner.df_filtered.to_csv(cleaned_file_names[i], index=False)
"""