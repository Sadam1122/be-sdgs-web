from scrappingSinta import WebScraper
from preprocessSinta import SintaPreprocessor
import pandas as pd
import datetime
from transformers import pipeline, AutoTokenizer
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Fungsi untuk memotong teks menggunakan tokenizer
def truncate_text(text, tokenizer, max_length=512):
    tokens = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
    return tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)

def classify_sdgs(text):
    tokenizer = AutoTokenizer.from_pretrained("Zaniiiii/sdgs")
    classifier = pipeline("text-classification", model="Zaniiiii/sdgs", return_all_scores=True)
    truncated_text = truncate_text(text, tokenizer)
    results = classifier(truncated_text)
    labels = [result['label'] for result in results[0] if result['score'] > 0.5]
    return labels if labels else None

def run_sinta_main():
    try:
        sinta_credentials = ("suryoadhiwibowo@telkomuniversity.ac.id", "Bangkit2023!")
        elsevier_credentials = ("hamdanazani@student.telkomuniversity.ac.id", "dayak1352")
        sinta_url = 'https://sinta.kemdikbud.go.id/affiliations/profile/1093?page=1&view=scopus'
        num_pages = 2

        scraper = WebScraper()
        scraper.run(sinta_credentials, elsevier_credentials, sinta_url, num_pages)

        current_date = datetime.date.today()
        end_day, end_month, end_year = current_date.day, current_date.month, current_date.year
        file_sinta = f'./sinta/storage/result/scrappingSinta/crawleddSinta{end_day}-{end_month}-{end_year}.csv'
        preprocessor = SintaPreprocessor(file_sinta)
        processed_df = preprocessor.preprocess()
        file_result = f'.preProcessSinta{end_day}-{end_month}-{end_year}'
        print("save_result ===========================================================================================" )
        preprocessor.save_result(file_result)        
        ## Klasifikasi
        df_final = pd.read_json(f'./sinta/storage/result/preprocessSinta/{file_result}.json')
        print(df_final.info())
        df = pd.read_json("../hasil_akhir.json")
        classifier = pipeline("text-classification", model="Zaniiiii/sdgs", return_all_scores=True)

        tokenizer = AutoTokenizer.from_pretrained("Zaniiiii/sdgs")
        df_final['Sdgs'] = df_final['Abstrak'].apply(classify_sdgs)
        df_final["Source"] = "Sinta"
        print(df_final.info())
        print(df.info())

        # Check if all rows in df_final are already in df
        merged = df.merge(df_final, on='Judul', how='inner')
        if len(merged) == len(df_final):
            raise ValueError("No data added: All new crawled data already.")
        # If not all are duplicates, proceed with concatenation and save the updated dataframe
        df_combined = pd.concat([df, df_final])
        df_combined = df_combined.drop_duplicates(subset=['Judul'])
        df_combined.to_json("../hasil_akhir.json", orient='records')
        print(df_combined.info())
    except Exception as e:
        print("Terjadi error:", str(e))
        raise
run_sinta_main()
