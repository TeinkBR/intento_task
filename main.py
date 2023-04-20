# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from Intento_additional_task import DatasetCleaner

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