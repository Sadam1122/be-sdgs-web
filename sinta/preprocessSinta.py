import pandas as pd
import numpy as np
import re
from deep_translator import GoogleTranslator
import os
import time
from bs4 import BeautifulSoup


class SintaPreprocessor:
    def __init__(self, file_path):
        """Initialize the processor with the path to the CSV file."""
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
    
    def list_to_string(self, lst):
        """Converts a list of strings to a single string."""
        return ' '.join(lst)

    def cleaning_penulis(self, names):
        """Clean and format author names in 'penulis' column."""
        cleaned_names = []
        for name in names:
            parts = name.strip().split(',')
            if len(parts) == 2:
                cleaned_names.append(parts[1].strip() + ' ' + parts[0].strip())
            else:
                cleaned_names.append(parts[0].strip())
        return cleaned_names

    def cleaning_abstrak_tahap1(self, text):
        """Initial cleaning of abstract text (lowercase, remove '©')."""
        text = str(text).lower()
        text = re.sub(r'©.*', '', text)
        return text

    def cleaning_judul(self, text):
        """Clean the title (remove non-alphabetical characters and capitalize)."""
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = str(text).capitalize()
        return text
    
    def cleaningAbstrak(self, text, retries=3, backoff_factor=1.5, row_index=None, total_rows=None):
        """
        Clean and translate the abstract text.
        Retry mechanism for failed translations.
        """
        if row_index is not None and total_rows is not None:
            print(f"Processing row {row_index + 1}/{total_rows}")

        # Clean text
        text = str(text)
        text = BeautifulSoup(text, "html.parser").get_text()
        text = text.lower()
        text = re.sub(r'http\\S+', '', text)
        text = re.sub('(@\\w+|#\\w+)', '', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = re.sub("\\n", " ", text)
        text = re.sub('(s{2,})', ' ', text)

        # Translation process with retry logic
        translator = GoogleTranslator(source='en', target='id')
        translated_text = None
        attempt = 0

        while attempt < retries:
            try:
                translated_text = translator.translate(text)
                break
            except Exception as e:
                attempt += 1
                wait_time = backoff_factor ** attempt
                print(f"Translation attempt {attempt} failed: {e}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)  # Exponential backoff

        if translated_text is None:
            print(f"Skipping translation for row {row_index + 1}/{total_rows} due to repeated failures.")
            return text  # Optionally return the original text or an empty string

        translated_text = translated_text.lower()
        translated_text = re.sub(r'http\\S+', '', translated_text)
        translated_text = re.sub('(@\\w+|#\\w+)', '', translated_text)
        translated_text = re.sub('[^a-zA-Z]', ' ', translated_text)
        translated_text = re.sub("\\n", " ", translated_text)
        translated_text = re.sub('(s{2,})', ' ', translated_text)

        return translated_text

    def get_aspects(self, review):
        """
        Extract SDGs aspects from the review text by matching keywords.
        """
        review = str(review) if review else ''
        review_aspects = []

        aspects = {
            'SDGS1': ['Goal 1'],
            'SDGS2': ['Goal 2'],
            'SDGS3': ['Goal 3'],
            'SDGS4': ['Goal 4'],
            'SDGS5': ['Goal 5'],
            'SDGS6': ['Goal 6'],
            'SDGS7': ['Goal 7'],
            'SDGS8': ['Goal 8'],
            'SDGS9': ['Goal 9'],
            'SDGS10': ['Goal 10'],
            'SDGS11': ['Goal 11'],
            'SDGS12': ['Goal 12'],
            'SDGS13': ['Goal 13'],
            'SDGS14': ['Goal 14'],
            'SDGS15': ['Goal 15'],
            'SDGS16': ['Goal 16'],
            'SDGS17': ['Goal 17']
        }

        for aspect, keywords in aspects.items():
            for keyword in keywords:
                if re.search(fr'{re.escape(keyword)}', review):
                    review_aspects.append(aspect)
                    break
        return review_aspects

    def process_row(self, row, total_rows):
        """Process each row to clean the abstract text."""
        print(f"Processing row {row.name + 1}/{total_rows}")
        return self.cleaningAbstrak(row['abstrak'])

    def penulisGaAda(self, penulis, dosen_list):
        """Check if authors in 'penulis' are in the 'dosen_list'."""
        penulis_list = [nama.strip() for nama in penulis.split(',')]
        if set(penulis_list) & set(dosen_list):
            return False
        return True
    
    def gantiNama(self, penulis, benarkan_nama):
        """Replace incorrect author names with correct ones."""
        penulis_list = [nama.strip() for nama in penulis.split(',')]
        penulis_list_baru = [benarkan_nama.get(nama.strip(), np.nan) for nama in penulis_list]
        return ', '.join([nama if pd.notna(nama) else '0' for nama in penulis_list_baru])

    def penulisAda(self, penulis, dosen_list):
        """Check if any author in 'penulis' exists in 'dosen_list'."""
        penulis_list = [nama.strip() for nama in penulis.split(',')]
        if set(penulis_list) & set(dosen_list):
            return True
        return False

    def preprocess(self):
        """Main function to preprocess the data."""
        self.df = self.df[["Authors", "Title", "Year", "Abstract"]]
        self.df = self.df.rename(columns={'Title': 'judul', 'Authors': 'penulis', 'Abstract': 'abstrak', 'Year': 'tahun'})

        self.df['penulis'] = self.df['penulis'].apply(lambda x: x.replace('Save all to author list', ''))
        self.df['penulis'] = self.df['penulis'].apply(lambda x: x.split(';'))
        self.df['penulis'] = self.df['penulis'].apply(self.cleaning_penulis)
        self.df['penulis'] = self.df['penulis'].apply(lambda x: ', '.join(x))

        self.df = self.df.dropna()
        total_rows = len(self.df)

        self.df["judul"] = self.df["judul"].apply(self.cleaning_judul)

        self.df = self.df.rename(columns={'judul': 'Judul', 'penulis': 'Penulis', 'abstrak': 'Abstrak', 'tahun': 'Tahun'})

        # Read dosen data
        df1 = pd.read_csv('dataDosen.csv', sep=';')
        df1['NAMA LENGKAP'] = df1['NAMA LENGKAP'].str.title()
        dosen_list = df1['NAMA LENGKAP'].tolist()

        # Filter authors not in dosen_list
        df2 = self.df[self.df['Penulis'].apply(lambda penulis: self.penulisGaAda(penulis, dosen_list))]

        # Correct author names based on 'benarkanNama.txt'
        benarkan_nama = {}
        with open('benarkanNama.txt', 'r') as file:
            for line in file:
                parts = line.strip().split(', ')
                if len(parts) == 2:
                    salah, benar = parts
                    benarkan_nama[salah.strip()] = benar.strip()

        df2["Penulis"] = df2["Penulis"].apply(lambda penulis: self.gantiNama(penulis, benarkan_nama))
        df2['Penulis'] = df2['Penulis'].astype(str).apply(lambda x: ', '.join([item.strip() for item in x.split(',') if item.strip() != '0']))
        df2['Penulis'] = df2['Penulis'].replace(['', 'nan'], np.nan)
        df2.dropna(subset=['Penulis'], inplace=True)

        # Final dataframe with authors in dosen_list
        df_final = self.df[self.df['Penulis'].apply(lambda penulis: self.penulisAda(penulis, dosen_list))]

        # Concatenate final and filtered dataframes
        self.df = pd.concat([df_final, df2])

        # Process abstracts
        self.df['Abstrak'] = self.df.apply(lambda row: self.cleaningAbstrak(row['Abstrak'], row_index=row.name, total_rows=total_rows), axis=1)
        return self.df

    def save_result(self, file_name):
        """Save the final processed data as a JSON file."""
        if not os.path.exists('./sinta/storage/result/preprocessSinta'):
            os.makedirs('./sinta/storage/result/preprocessSinta')
        self.df.to_json(f'./sinta/storage/result/preprocessSinta/{file_name}.json', orient='records')

    def save_result_main(self, file_name):
        """Save the final processed data to a different location."""
        if not os.path.exists('./sinta/storage/result/preprocessSinta'):
            os.makedirs('./sinta/storage/result/preprocessSinta')
        self.df.to_json(f'./sinta/storage/result/preprocessSinta/{file_name}.json', orient='records')

    def save_result2(self, file_name):
        """Save the result to the upload folder."""
        if not os.path.exists('../upload'):
            os.makedirs('../upload')
        self.df.to_json(f'../upload/{file_name}.json', orient='records')
