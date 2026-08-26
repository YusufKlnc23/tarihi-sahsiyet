# tarihi-sahsiyet

Turk tarihi sahsiyetleri hakkinda PDF/TXT kaynaklardan uretilmis chunk'lara dayanarak cevap veren Gradio demo uygulamasi.

## Demo

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python app.py
```

Adres:

```text
http://127.0.0.1:7860
```

## Ortam Degiskenleri

Gercek degerleri `.env` dosyasinda veya sistem ortaminda tutun. GitHub'a gercek anahtar veya parola yuklemeyin.

```env
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/tarih_figures
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=gemini-3.5-flash
LLM_PROVIDER=gemini
```

## Veri Hazirlama

```powershell
tarih-analyze init-db
tarih-analyze load-figures data/figures.example.json
tarih-analyze ingest data/pdfs --force
```

PDF ve TXT kaynaklar GitHub'a eklenmez. `data/pdfs` ve `data/texts` sadece yerel kaynak klasorleridir.


