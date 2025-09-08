# MANTA (Multi-lingual Advanced NMF-based Topic Analysis)

[![PyPI version](https://badge.fury.io/py/manta-topic-modelling.svg)](https://badge.fury.io/py/manta-topic-modelling)
[![PyPI version](https://img.shields.io/pypi/v/manta-topic-modelling)](https://badge.fury.io/py/manta-topic-modelling)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Negatif Olmayan Matris Faktörizasyonu (NMF) kullanarak hem İngilizce hem de Türkçe metin işlemeyi destekleyen kapsamlı bir konu modelleme sistemi. Gelişmiş tokenizasyon teknikleri, çoklu NMF algoritmaları ve zengin görselleştirme özelliklerine sahiptir.

## Hızlı Başlangıç
### Geliştirme için Yerel Kurulum
Uygulamayı geliştirme için yerel olarak derlemek ve çalıştırmak için:
Önce depoyu klonlayın:
```bash
git clone https://github.com/emirkyz/manta.git
```
Klonladıktan sonra, proje dizinine gidin ve sanal ortam oluşturun:

```bash
cd manta
python -m venv .venv
source .venv/bin/activate  # Windows'ta: .venv\Scripts\activate
```
Ardından, gerekli bağımlılıkları yükleyin. Eğer `pip` yüklüyse, şunu çalıştırabilirsiniz:
```bash
pip install -e .
```
veya `uv` yüklüyse, şunu kullanabilirsiniz:
```bash
uv pip install -e .
```
### PyPI'den Kurulum
```bash
pip install manta-topic-modelling
```

Bundan sonra uygulamayı içe aktarıp kullanabilirsiniz.

### Komut Satırı Kullanımı
```bash
# Türkçe metin analizi
manta-topic-modelling analyze data.csv --column text --language TR --topics 5

# Lemmatizasyon ve görselleştirmelerle İngilizce metin analizi
manta-topic-modelling analyze data.csv --column content --language EN --topics 10 --lemmatize --wordclouds --excel

# Türkçe metin için özel tokenizer
manta-topic-modelling analyze reviews.csv --column review_text --language TR --topics 8 --tokenizer bpe --wordclouds

# Uygulama adı ve ülkeye göre filtreleme
manta-topic-modelling analyze reviews.csv --column REVIEW --language TR --topics 5 --filter-app MyApp --filter-country TR

# Özel filtreleme sütunları
manta-topic-modelling analyze data.csv --column text --language TR --topics 5 --filter-app-column APP_ID --filter-country-column REGION

# Daha hızlı işleme için emoji işlemeyi devre dışı bırak
manta-topic-modelling analyze data.csv --column text --language EN --topics 5 --emoji-map False
```

### Python API Kullanımı
```python
from manta import run_topic_analysis

# Basit konu modelleme
results = run_topic_analysis(
    filepath="data.csv",
    column="review_text",
    language="EN",
    topic_count=5,
    lemmatize=True
)

# Türkçe metin analizi
results = run_topic_analysis(
    filepath="turkish_reviews.csv", 
    column="yorum_metni",
    language="TR",
    topic_count=8,
    tokenizer_type="bpe",
    generate_wordclouds=True
)
```

## Sonuç Yapısı
```
{
"state": Analizin durumu, "success" veya "error",
"message": Analiz sonucu hakkında mesaj,
"data_name": Giriş veri dosyasının adı,
"topic_word_scores": Konuları ve skorlarıyla birlikte en önemli kelimelerini içeren JSON nesnesi,
"topic_doc_scores": Konuları ve skorlarıyla birlikte en önemli dokümanlarını içeren JSON nesnesi,
"coherence_scores": Her konu için tutarlılık skorlarını içeren JSON nesnesi,
"topic_dist_img": Eğer `gen_topic_distribution` True ise konu dağılım grafiğinin Matplotlib plt nesnesi,
"topic_document_counts": Konu başına doküman sayısı,
}
```
```
Örneğin:
{
  "state": "success",
  "message": "Analiz başarıyla tamamlandı",
  "data_name": "reviews.csv",
  "topic_word_scores": {
    "topic_0": {
        "kelime1": 0.15,
        "kelime2": 0.12,
        "kelime3": 0.10
        }
    },
  "topic_doc_scores":{
          "topic_0": [
                {
                    "document": "Örnek doküman metni...",
                    "score": 0.78
                }
            ],
    }
  "coherence_scores": {
        "gensim": {
           "umass_average": -1.4328882390292266,
            "umass_per_topic": {
                "topic_0": -1.4328882390292266,
                "topic_1": -1.1234567890123456,
                "topic_2": -0.9876543210987654
                }
        }
    },
  "topic_dist_img": "<matplotlib plot nesnesi>",
  "topic_document_counts": [____]
}
```

## Paket Yapısı

```
manta/
├── _functions/
│   ├── common_language/          # Diller arası paylaşılan fonksiyonellik
│   │   ├── emoji_processor.py    # Emoji işleme yardımcı araçları
│   │   └── topic_analyzer.py     # Diller arası konu analizi
│   ├── english/                  # İngilizce metin işleme modülleri
│   │   ├── english_entry.py             # İngilizce metin işleme giriş noktası
│   │   ├── english_preprocessor.py      # Metin temizleme ve ön işleme
│   │   ├── english_vocabulary.py        # Kelime dağarcığı oluşturma
│   │   ├── english_text_encoder.py      # Metin-sayısal dönüştürme
│   │   ├── english_topic_analyzer.py    # Konu çıkarım yardımcı araçları
│   │   ├── english_topic_output.py      # Konu görselleştirme ve çıktı
│   │   └── english_nmf_core.py          # İngilizce için NMF implementasyonu
│   ├── nmf/                      # NMF algoritma implementasyonları
│   │   ├── nmf_orchestrator.py          # Ana NMF arayüzü
│   │   ├── nmf_initialization.py        # Matris başlatma stratejileri
│   │   ├── nmf_basic.py                 # Standart NMF algoritması
│   │   ├── nmf_projective_basic.py      # Temel projektif NMF
│   │   └── nmf_projective_enhanced.py   # Gelişmiş projektif NMF
│   ├── tfidf/                    # TF-IDF hesaplama modülleri
│   │   ├── tfidf_english_calculator.py  # İngilizce TF-IDF implementasyonu
│   │   ├── tfidf_turkish_calculator.py  # Türkçe TF-IDF implementasyonu
│   │   ├── tfidf_tf_functions.py        # Terim frekansı fonksiyonları
│   │   ├── tfidf_idf_functions.py       # Ters doküman frekansı fonksiyonları
│   │   └── tfidf_bm25_turkish.py        # Türkçe için BM25 implementasyonu
│   └── turkish/                  # Türkçe metin işleme modülleri
│       ├── turkish_entry.py             # Türkçe metin işleme giriş noktası
│       ├── turkish_preprocessor.py      # Türkçe metin temizleme
│       ├── turkish_tokenizer_factory.py # Tokenizer oluşturma ve eğitimi
│       ├── turkish_text_encoder.py      # Metin-sayısal dönüştürme
│       └── turkish_tfidf_generator.py   # TF-IDF matris üretimi
├── utils/                        # Yardımcı araçlar
│   ├── coherence_score.py              # Konu tutarlılığı değerlendirme
│   ├── combine_number_suffix.py         # Sayı ve sonek birleştirme yardımcı araçları
│   ├── distance_two_words.py           # Kelime mesafesi hesaplama
│   ├── export_excel.py                 # Excel dışa aktarma fonksiyonelliği
│   ├── gen_cloud.py                    # Kelime bulutu üretimi
│   ├── hierarchy_nmf.py                # Hiyerarşik NMF yardımcı araçları
│   ├── image_to_base.py                # Görüntüden base64 dönüştürme
│   ├── save_doc_score_pair.py          # Doküman-skor çifti kaydetme yardımcı araçları
│   ├── save_topics_db.py               # Konu veritabanı kaydetme
│   ├── save_word_score_pair.py         # Kelime-skor çifti kaydetme yardımcı araçları
│   ├── topic_dist.py                   # Konu dağılım grafikleri
│   ├── umass_test.py                   # UMass tutarlılık testi
│   ├── visualizer.py                   # Genel görselleştirme yardımcı araçları
│   ├── word_cooccurrence.py            # Kelime birlikte bulunma analizi
│   └── other/                           # Ek yardımcı fonksiyonlar
├── cli.py                        # Komut satırı arayüzü
├── standalone_nmf.py             # Temel NMF implementasyonu
└── __init__.py                   # Paket başlatma ve public API
```

## Kurulum

### PyPI'den (Önerilen)
```bash
pip install manta-topic-modelling
```

### Kaynak Koddan (Geliştirme)
1. Depoyu klonlayın:
```bash
git clone https://github.com/emirkyz/manta.git
cd manta
```

2. Sanal ortam oluşturun:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows'ta: .venv\Scripts\activate
```

3. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

### Komut Satırı Arayüzü

Paket, `analyze` alt komutu ile `manta-topic-modelling` komutu sağlar:

```bash
# Temel kullanım
manta-topic-modelling analyze data.csv --column text --language TR --topics 5

# Tüm seçeneklerle gelişmiş kullanım
manta-topic-modelling analyze reviews.csv \
  --column review_text \
  --language EN \
  --topics 10 \
  --words-per-topic 20 \
  --nmf-method opnmf \
  --lemmatize \
  --wordclouds \
  --excel \
  --topic-distribution \
  --output-name my_analysis
```

#### Komut Satırı Seçenekleri

**Gerekli Argümanlar:**
- `filepath`: Giriş CSV veya Excel dosyasının yolu
- `--column, -c`: Metin verilerini içeren sütun adı
- `--language, -l`: Dil (Türkçe için "TR", İngilizce için "EN")

**İsteğe Bağlı Argümanlar:**
- `--topics, -t`: Çıkarılacak konu sayısı (varsayılan: 5)
- `--output-name, -o`: Çıktı dosyaları için özel ad (varsayılan: otomatik oluşturulan)
- `--tokenizer`: Türkçe için tokenizer türü ("bpe" veya "wordpiece", varsayılan: "bpe")
- `--nmf-method`: NMF algoritması ("nmf" veya "opnmf", varsayılan: "nmf")
- `--words-per-topic`: Konu başına en önemli kelime sayısı (varsayılan: 15)
- `--lemmatize`: İngilizce metin için lemmatizasyon uygula
- `--emoji-map`: Emoji işleme ve eşlemeyi etkinleştir (varsayılan: True). Devre dışı bırakmak için --emoji-map False kullanın
- `--wordclouds`: Kelime bulutu görselleştirmeleri oluştur
- `--excel`: Sonuçları Excel formatında dışa aktar
- `--topic-distribution`: Konu dağılım grafikleri oluştur
- `--separator`: CSV ayırıcı karakteri (varsayılan: "|")
- `--filter-app`: Belirli uygulama adına göre verileri filtrele
- `--filter-app-column`: Uygulama filtrelemesi için sütun adı (varsayılan: "PACKAGE_NAME")
- `--filter-country`: Ülke koduna göre verileri filtrele (örn. TR, US, GB)
- `--filter-country-column`: Ülke filtrelemesi için sütun adı (varsayılan: "COUNTRY")

### Python API

```python
from manta import run_topic_analysis

# Temel İngilizce metin analizi
results = run_topic_analysis(
    filepath="data.csv",
    column="review_text",
    language="EN",
    topic_count=5,
    lemmatize=True,
    generate_wordclouds=True,
    export_excel=True
)

# Filtreleme ile gelişmiş Türkçe metin analizi
results = run_topic_analysis(
    filepath="turkish_reviews.csv",
    column="yorum_metni",
    language="TR",
    topic_count=10,
    words_per_topic=15,
    tokenizer_type="bpe",
    nmf_method="nmf",
    generate_wordclouds=True,
    export_excel=True,
    topic_distribution=True,
    filter_app=True,
    data_filter_options={
        "filter_app_name": "MyApp",
        "filter_app_column": "APP_NAME",
        "filter_app_country": "TR",
        "filter_app_country_column": "COUNTRY_CODE"
    }
)
```

#### API Parametreleri

**Gerekli:**
- `filepath` (str): Giriş CSV veya Excel dosyasının yolu
- `column` (str): Metin verilerini içeren sütun adı

**İsteğe Bağlı:**
- `separator` (str): CSV ayırıcı karakteri (varsayılan: ",")
- `language` (str): Türkçe için "TR", İngilizce için "EN" (varsayılan: "EN")
- `topic_count` (int): Çıkarılacak konu sayısı (varsayılan: 5)
- `nmf_method` (str): "nmf", "pnmf", veya "nmtf" algoritma çeşidi (varsayılan: "nmf")
- `lemmatize` (bool): İngilizce için lemmatizasyon uygula (varsayılan: False)
- `tokenizer_type` (str): Türkçe için "bpe" veya "wordpiece" (varsayılan: "bpe")
- `words_per_topic` (int): Konu başına gösterilecek en önemli kelime (varsayılan: 15)
- `word_pairs_out` (bool): Kelime çiftleri çıktısı oluştur (varsayılan: True)
- `generate_wordclouds` (bool): Kelime bulutu görselleştirmeleri oluştur (varsayılan: True)
- `export_excel` (bool): Sonuçları Excel'e dışa aktar (varsayılan: True)
- `topic_distribution` (bool): Dağılım grafikleri oluştur (varsayılan: True)
- `filter_app` (bool): Uygulama filtrelemesini etkinleştir (varsayılan: False)
- `data_filter_options` (dict): Gelişmiş filtreleme seçenekleri (tüm anahtarlar varsayılan olarak boş string):
  - `filter_app_name` (str): Filtreleme için uygulama adı
  - `filter_app_column` (str): Uygulama filtrelemesi için sütun adı (varsayılan: "PACKAGE_NAME")
  - `filter_app_country` (str): Ülke koduna göre filtreleme (büyük/küçük harf duyarsız)
  - `filter_app_country_column` (str): Ülke filtrelemesi için sütun adı (varsayılan: "COUNTRY")
- `emoji_map` (bool): Emoji işleme ve eşlemeyi etkinleştir (varsayılan: False)
- `output_name` (str): Özel çıktı dizini adı (varsayılan: otomatik oluşturulan)
- `save_to_db` (bool): Veritabanına kalıcı olarak kaydet (varsayılan: False)
- `output_dir` (str): Çıktılar için temel dizin (varsayılan: mevcut çalışma dizini)

## Çıktılar

Analiz, `Output/` dizininde (çalışma zamanında oluşturulan) analizinizin adını taşıyan alt dizinde organize edilmiş çeşitli çıktılar oluşturur:

- **Konu-Kelime Excel Dosyası**: Her konu için en önemli kelimeleri ve skorlarını içeren `.xlsx` dosyası
- **Kelime Bulutları**: Her konu için kelime bulutlarının PNG görüntüleri (`generate_wordclouds=True` ise)
- **Konu Dağılım Grafiği**: Dokümanların konulara göre dağılımını gösteren grafik (`topic_distribution=True` ise)
- **Tutarlılık Skorları**: Konular için tutarlılık skorlarını içeren JSON dosyası
- **En İyi Dokümanlar**: Her konu için en temsili dokümanları listeleyen JSON dosyası

## Özellikler

- **Çok Dilli Destek**: Hem Türkçe hem de İngilizce metinler için optimize edilmiş işleme
- **Gelişmiş Tokenizasyon**: Türkçe için BPE ve WordPiece tokenizer'ları, İngilizce için geleneksel tokenizasyon
- **Çoklu NMF Algoritmaları**: Standart NMF ve Ortogonal Projektif NMF (OPNMF)
- **Zengin Görselleştirmeler**: Kelime bulutları ve konu dağılım grafikleri
- **Esnek Dışa Aktarma**: Excel ve JSON dışa aktarma formatları
- **Tutarlılık Değerlendirmesi**: Yerleşik konu tutarlılığı skorlaması
- **Metin Ön İşleme**: Dile özgü metin temizleme ve ön işleme

## Gereksinimler

- Python 3.9+
- Bağımlılıklar paket ile otomatik olarak yüklenir

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın.

## Katkıda Bulunma

Katkılar memnuniyetle karşılanır! Lütfen Pull Request göndermekten çekinmeyin.

## Destek

Sorunlar ve sorular için lütfen [GitHub deposunda](https://github.com/emirkyz/manta/issues?q=is%3Aissue) bir issue açın