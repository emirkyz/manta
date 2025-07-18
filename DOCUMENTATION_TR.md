# MANTA (Multi-lingual Advanced NMF-based Topic Analysis) - Detaylı Türkçe Dokümantasyon

## İçindekiler
1. [Proje Genel Bakış](#proje-genel-bakış)
2. [Mimari Genel Bakış](#mimari-genel-bakış)
3. [Ana Fonksiyon Dokümantasyonu](#ana-fonksiyon-dokümantasyonu)
4. [İş Akışı Süreci](#iş-akışı-süreci)
5. [Modül Yapısı](#modül-yapısı)
6. [Konfigürasyon Seçenekleri](#konfigürasyon-seçenekleri)
7. [Çıktı Dosyaları](#çıktı-dosyaları)
8. [Kullanım Örnekleri](#kullanım-örnekleri)
9. [Sorun Giderme](#sorun-giderme)

---

## Proje Genel Bakış

**MANTA (Multi-lingual Advanced NMF-based Topic Analysis)**, metin belgelerinden anlamlı konuları çıkarmak için **Negatif Olmayan Matris Faktörizasyonu (Non-negative Matrix Factorization - NMF)** kullanan kapsamlı bir konu modelleme aracıdır. Sistem hem **Türkçe** hem de **İngilizce** dilleri destekler ve ham metin verilerinden görselleştirilmiş konu sonuçlarına kadar uçtan uca işleme sağlar.

### Temel Özellikler
- **İki Dilli Destek**: Hem Türkçe hem de İngilizce metin işleme desteği
- **Çoklu Tokenizasyon Yöntemleri**: Türkçe için BPE (Byte-Pair Encoding) ve WordPiece
- **Esnek NMF Algoritmaları**: Standart NMF ve Ortogonal Projektif NMF (OPNMF)
- **Zengin Çıktı Üretimi**: Kelime bulutları, konu dağılımları, Excel raporları ve tutarlılık skorları
- **Veritabanı Entegrasyonu**: Kalıcı depolama için SQLite veritabanları
- **Kapsamlı Ön İşleme**: Metin temizleme, tokenizasyon ve TF-IDF vektörizasyonu

### Kullanım Alanları
- **Akademik Araştırma**: Araştırma makaleleri, tezler ve akademik metinlerin analizi
- **App Store Analizi**: Kullanıcı yorumları ve geri bildirimlerin işlenmesi
- **Sosyal Medya Madenciliği**: Sosyal medya gönderilerinden konu çıkarımı
- **Doküman Kümeleme**: Büyük doküman koleksiyonlarının konulara göre organize edilmesi
- **İçerik Analizi**: Metinsel verilerdeki tematik kalıpların anlaşılması

---

## Mimari Genel Bakış

```
┌─────────────────────────────────────────────────────────────────┐
│                        GİRİŞ VERİSİ                            │
│                    (CSV/Excel Dosyaları)                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     VERİ ÖN İŞLEME                             │
│  ┌─────────────────┐              ┌─────────────────┐          │
│  │   Türkçe Yol    │              │  İngilizce Yol  │          │
│  │  - Metin Temizl.│              │ - Sözlük        │          │
│  │  - Tokenizasyon │              │   Oluşturma     │          │
│  │  - BPE/WordPiece│              │ - Lemmatizasyon │          │
│  └─────────────────┘              └─────────────────┘          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TF-IDF VEKTÖRİZASYONU                       │
│               Metni sayısal matrise dönüştür                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NMF AYRIŞTIRIMASI                           │
│              W (Doküman-Konu) × H (Konu-Kelime)               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    KONU ANALİZİ                                │
│  - Konu Kelime Çıkarımı     - Doküman Sınıflandırması          │
│  - Tutarlılık Skorlama      - Temsili Dokümanlar               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ÇIKTI ÜRETİMİ                               │
│  - Kelime Bulutları     - Excel Raporları                      │
│  - Dağılım Grafikleri   - JSON Verisi                          │
│  - Veritabanı Depolama  - Tutarlılık Metrikleri                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Ana Fonksiyon Dokümantasyonu

### 1. process_turkish_file()

Türkçe metin verilerini tam ön işleme hattından geçirir.

```python
def process_turkish_file(df, desired_columns: str, tokenizer=None, tokenizer_type=None):
    """
    Özelleştirilmiş Türkçe NLP hattı ile Türkçe metin verilerini işler.
    
    Args:
        df (pandas.DataFrame): Türkçe metin içeren giriş dataframe'i
        desired_columns (str): Analiz edilecek metni içeren sütun adı
        tokenizer (isteğe bağlı): Önceden eğitilmiş tokenizer nesnesi
        tokenizer_type (str): "bpe" veya "wordpiece"
    
    Returns:
        tuple: (tdm, sozluk, sayisal_veri, tokenizer)
            - tdm: TF-IDF doküman-terim matrisi
            - sozluk: Kelime dağarcığı listesi
            - sayisal_veri: Dokümanların sayısal temsili
            - tokenizer: Eğitilmiş tokenizer nesnesi
    """
```

**İşlem Akışı:**
1. **Metin Temizleme** (`metin_temizle_turkish`): Gürültüyü, özel karakterleri kaldırır ve metni normalleştirir
2. **Tokenizer Başlatma**: Sağlanmadıysa BPE veya WordPiece tokenizer oluşturur
3. **Tokenizer Eğitimi**: Kelime dağarcığı oluşturmak için tokenizer'ı korpus üzerinde eğitir
4. **Kelime Dağarcığı Çıkarımı**: Eğitilmiş tokenizer'dan kelime tokenlarını çıkarır
5. **Sayısal Dönüştürme**: Tokenizer kullanarak metini sayısal vektörlere dönüştürür
6. **TF-IDF Üretimi**: Ağırlıklı doküman-terim matrisi oluşturur

### 2. process_english_file()

İngilizce metin işlemeyi lemmatizasyon ve sözlük tabanlı yaklaşımlarla ele alır.

```python
def process_english_file(df, desired_columns: str, lemmatize: bool):
    """
    İngilizce NLP hattı ile İngilizce metin verilerini işler.
    
    Args:
        df (pandas.DataFrame): İngilizce metin içeren giriş dataframe'i
        desired_columns (str): Analiz edilecek metni içeren sütun adı
        lemmatize (bool): Lemmatizasyon uygulanıp uygulanmayacağı
    
    Returns:
        tuple: (tdm, sozluk, sayisal_veri)
            - tdm: TF-IDF doküman-terim matrisi (sayisal_veri ile aynı)
            - sozluk: Terimlerin sözlüğü/kelime dağarcığı
            - sayisal_veri: TF-IDF ağırlıklı doküman-terim matrisi
    """
```

**İşlem Akışı:**
1. **Sözlük Oluşturma** (`sozluk_yarat`): İsteğe bağlı lemmatizasyon ile kelime dağarcığı oluşturur
2. **TF-IDF Hesaplama** (`tf_idf_english`): TF-IDF ağırlıklarını doğrudan hesaplar
3. **Matris Hazırlama**: Son doküman-terim matrisini hazırlar

### 3. process_file()

Tüm konu modelleme hattını koordine eden ana orkestrasyon fonksiyonu.

```python
def process_file(
    filepath: str,
    table_name: str,
    desired_columns: str,
    desired_topic_count: int,
    LEMMATIZE: bool,
    N_TOPICS: int,
    tokenizer=None,
    LANGUAGE="TR",
    tokenizer_type="bpe",
    gen_topic_distribution=True,
    separator=",",
    nmf_type="nmf"
) -> dict:
    """
    Dosya girişinden sonuçlara kadar tam konu modelleme hattı.
    
    Args:
        filepath (str): Giriş CSV/Excel dosyasının yolu
        table_name (str): Bu analiz çalışması için benzersiz tanımlayıcı
        desired_columns (str): Metin verilerini içeren sütun adı
        desired_topic_count (int): Çıkarılacak konu sayısı
        LEMMATIZE (bool): Lemmatizasyonu etkinleştir (ağırlıklı olarak İngilizce için)
        N_TOPICS (int): Konu başına gösterilecek üst kelime sayısı
        tokenizer (isteğe bağlı): Önceden başlatılmış tokenizer
        LANGUAGE (str): Türkçe için "TR", İngilizce için "EN"
        tokenizer_type (str): Türkçe için "bpe" veya "wordpiece"
        gen_topic_distribution (bool): Konu dağılım grafikleri oluştur
        separator (str): CSV ayırıcı karakteri
        nmf_type (str): "nmf" veya "opnmf" algoritma seçimi
    
    Returns:
        dict: Şunları içeren sonuçlar:
            - state: "SUCCESS" veya "FAILURE"
            - message: Durum mesajı
            - data_name: Analiz tanımlayıcısı
            - topic_word_scores: Konu-kelime ilişkilendirmeleri
    """
```

**Tam İşlem Akışı:**

1. **Ortam Kurulumu**
   - Gerekli dizinleri oluşturur (`instance/`, `Output/`, `tfidf/`)
   - Konular ve ana veri için SQLite veritabanlarını başlatır

2. **Veri Yükleme ve Temizleme**
   ```python
   # CSV dosyaları için özel işleme
   if filepath.endswith(".csv"):
       # Sorunlu karakterleri değiştir
       data = data.replace("|", ";")
       data = data.replace("\\t", "")
       data = data.replace("\\x00", "")
   ```
   - Kodlama işleme ile CSV/Excel dosyalarını yükler
   - Verileri filtreler (örn. ülkeye özgü filtreleme)
   - Tekrarları ve boş değerleri kaldırır

3. **Veritabanı Depolamasi**
   - İşlenmiş veriyi SQLite veritabanında depolar
   - Belirtilen `table_name` ile tablo oluşturur

4. **Dile Özgü İşleme**
   ```python
   if LANGUAGE == "TR":
       tdm, sozluk, sayisal_veri, tokenizer = process_turkish_file(...)
   elif LANGUAGE == "EN":
       tdm, sozluk, sayisal_veri = process_english_file(...)
   ```

5. **NMF Ayrıştırması**
   ```python
   W, H = run_nmf(
       num_of_topics=int(desired_topic_count),
       sparse_matrix=tdm,
       norm_thresh=0.005,
       nmf_method=nmf_type
   )
   ```

6. **Konu Analizi ve Çıktı Üretimi**
   - Her konu için baskın kelimeleri ve dokümanları çıkarır
   - Tutarlılık skorlarını hesaplar
   - Kelime bulutları ve dağılım grafikleri oluşturur
   - Sonuçları Excel formatında dışa aktarır

### 4. run_standalone_nmf()

Basitleştirilmiş arayüz sağlayan ana giriş noktası fonksiyonu.

```python
def run_standalone_nmf(filepath, table_name, desired_columns, options):
    """
    NMF konu modellemesi için basitleştirilmiş giriş noktası.
    
    Args:
        filepath (str): Giriş dosyasının yolu
        table_name (str): Analiz tanımlayıcısı
        desired_columns (str): Metin sütunu adı
        options (dict): Şunları içeren konfigürasyon sözlüğü:
            - LEMMATIZE: bool
            - N_TOPICS: int (konu başına kelime)
            - DESIRED_TOPIC_COUNT: int
            - tokenizer_type: str
            - nmf_type: str
            - LANGUAGE: str
            - separator: str
            - gen_topic_distribution: bool
    
    Returns:
        dict: Zamanlama bilgisiyle birlikte işlem sonuçları
    """
```

---

## İş Akışı Süreci

### Detaylı Adım Adım İşlem

#### Faz 1: Veri Hazırlama
```
Giriş Dosyası → Veri Yükleme → Temizleme → Tekrar Kaldırma → Veritabanı Depolama
```

#### Faz 2: Metin İşleme (Dile Bağımlı)

**Türkçe Yolu:**
```
Ham Metin → Metin Temizleme → Tokenizer Eğitimi → Kelime Dağarcığı Oluşturma → 
Sayısal Dönüştürme → TF-IDF Matrisi
```

**İngilizce Yolu:**
```
Ham Metin → Sözlük Oluşturma → Lemmatizasyon (isteğe bağlı) → 
TF-IDF Hesaplama → Matris Hazırlama
```

#### Faz 3: Konu Modelleme
```
TF-IDF Matrisi → NMF Ayrıştırması → W Matrisi (Doküman-Konu) + 
H Matrisi (Konu-Kelime) → Konu Analizi
```

#### Faz 4: Sonuçlar ve Görselleştirme
```
Konu Matrisleri → Kelime Çıkarımı → Tutarlılık Hesaplama → 
Kelime Bulutları → Dağılım Grafikleri → Excel Dışa Aktarma → JSON Depolama
```

---

## Modül Yapısı

### _functions/ Dizini

#### common_language/
- **`emoji_processor.py`**: Emoji işleme yardımcı araçları
- **`topic_analyzer.py`**: Diller arası konu analizi

#### english/
- **`english_entry.py`**: İngilizce metin işleme giriş noktası
- **`english_preprocessor.py`**: Metin temizleme ve ön işleme
- **`english_vocabulary.py`**: Kelime dağarcığı oluşturma
- **`english_text_encoder.py`**: Metin-sayısal dönüştürme
- **`english_topic_analyzer.py`**: Konu çıkarım yardımcı araçları
- **`english_topic_output.py`**: Konu görselleştirme ve çıktı
- **`english_nmf_core.py`**: İngilizce için NMF implementasyonu

#### turkish/
- **`turkish_entry.py`**: Türkçe metin işleme giriş noktası
- **`turkish_preprocessor.py`**: Türkçe metin temizleme ve normalleştirme
- **`turkish_tokenizer_factory.py`**: Tokenizer oluşturma ve eğitimi (BPE/WordPiece)
- **`turkish_text_encoder.py`**: Türkçe metin sayısal dönüştürme
- **`turkish_tfidf_generator.py`**: Türkçe için TF-IDF üretimi

#### nmf/
- **`nmf_orchestrator.py`**: Ana NMF orkestrasyonu fonksiyonu
- **`nmf_basic.py`**: Standart NMF implementasyonu
- **`nmf_projective_basic.py`**: Temel projektif NMF
- **`nmf_projective_enhanced.py`**: Gelişmiş projektif NMF
- **`nmf_initialization.py`**: Matris başlatma stratejileri

#### tfidf/
- **`tfidf_english_calculator.py`**: İngilizce TF-IDF hesaplama
- **`tfidf_turkish_calculator.py`**: Türkçe TF-IDF üretimi
- **`tfidf_tf_functions.py`**: Terim frekansı hesaplama fonksiyonları
- **`tfidf_idf_functions.py`**: Ters doküman frekansı fonksiyonları
- **`tfidf_bm25_turkish.py`**: Türkçe için BM25 implementasyonu

### utils/ Dizini

- **`coherence_score.py`**: Konu tutarlılığı hesaplama
- **`combine_number_suffix.py`**: Sayı ve sonek birleştirme yardımcı araçları
- **`distance_two_words.py`**: Kelime mesafesi hesaplama
- **`export_excel.py`**: Excel raporu üretimi
- **`gen_cloud.py`**: Kelime bulutu üretimi
- **`hierarchy_nmf.py`**: Hiyerarşik NMF yardımcı araçları
- **`image_to_base.py`**: Görüntüden base64 dönüştürme
- **`save_doc_score_pair.py`**: Doküman-konu skor kalıcılığı
- **`save_topics_db.py`**: Konu veritabanı kaydetme
- **`save_word_score_pair.py`**: Kelime-skor çifti kaydetme yardımcı araçları
- **`topic_dist.py`**: Konu dağılımı görselleştirme
- **`umass_test.py`**: UMass tutarlılık testi
- **`visualizer.py`**: Genel görselleştirme yardımcı araçları
- **`word_cooccurrence.py`**: Kelime birlikte bulunma analizi
- **`other/`**: Ek yardımcı fonksiyonlar

---

## Konfigürasyon Seçenekleri

### Temel Parametreler

| Parametre | Tür | Açıklama | Varsayılan | Seçenekler |
|-----------|-----|----------|------------|------------|
| `LANGUAGE` | str | Metin dili | "TR" | "TR", "EN" |
| `desired_topic_count` | int | Çıkarılacak konu sayısı | 5 | 2-50+ |
| `N_TOPICS` | int | Konu başına gösterilecek kelime | 15 | 5-30 |
| `LEMMATIZE` | bool | Lemmatizasyonu etkinleştir (İngilizce) | True | True, False |
| `tokenizer_type` | str | Tokenizer türü (Türkçe) | "bpe" | "bpe", "wordpiece" |
| `nmf_type` | str | NMF algoritması | "nmf" | "nmf", "opnmf" |
| `separator` | str | CSV dosya ayırıcısı | "," | ",", ";", "\\t" |

### Gelişmiş Seçenekler

| Parametre | Tür | Açıklama | Varsayılan |
|-----------|-----|----------|------------|
| `gen_topic_distribution` | bool | Dağılım grafikleri oluştur | True |
| `gen_cloud` | bool | Kelime bulutları oluştur | True |
| `save_excel` | bool | Excel'e dışa aktar | True |
| `word_pairs_out` | bool | Kelime birlikte bulunma hesapla | False |
| `norm_thresh` | float | NMF normalleştirme eşiği | 0.005 |

### Örnek Konfigürasyon

```python
seçenekler = {
    "LEMMATIZE": True,
    "N_TOPICS": 15,
    "DESIRED_TOPIC_COUNT": 8,
    "tokenizer_type": "bpe",
    "nmf_type": "opnmf",
    "LANGUAGE": "TR",
    "separator": ";",
    "gen_cloud": True,
    "save_excel": True,
    "word_pairs_out": False,
    "gen_topic_distribution": True
}
```

---

## Çıktı Dosyaları

### Dizin Yapısı
```
Output/
└── {table_name}/
    ├── {table_name}_topics.xlsx              # Konu-kelime matrisi
    ├── {table_name}_coherence_scores.json    # Tutarlılık metrikleri
    ├── {table_name}_document_dist.png        # Konu dağılım grafiği
    ├── {table_name}_wordcloud_scores.json    # Kelime bulutu verisi
    ├── top_docs_{table_name}.json            # Temsili dokümanlar
    └── wordclouds/
        ├── Konu 00.png                       # 0. konu için kelime bulutu
        ├── Konu 01.png                       # 1. konu için kelime bulutu
        └── ...
```

### Dosya Açıklamaları

#### Excel Raporu (`{table_name}_topics.xlsx`)
- **Konu Sayfaları**: Her konu kendi çalışma sayfasını alır
- **Kelime Skorları**: Önem skorlarıyla birlikte üst kelimeler
- **Doküman Referansları**: Her konu için temsili dokümanlar

#### Tutarlılık Skorları (`{table_name}_coherence_scores.json`)
```json
{
    "topic_0": {
        "coherence_score": 0.45,
        "top_words": ["kelime1", "kelime2", "kelime3"]
    },
    "topic_1": {
        "coherence_score": 0.52,
        "top_words": ["kelime4", "kelime5", "kelime6"]
    }
}
```

#### Kelime Bulutu Verisi (`{table_name}_wordcloud_scores.json`)
```json
{
    "topic_0": {
        "kelime1": 0.15,
        "kelime2": 0.12,
        "kelime3": 0.10
    }
}
```

#### Temsili Dokümanlar (`top_docs_{table_name}.json`)
```json
{
    "topic_0": [
        {
            "document": "Örnek doküman metni...",
            "score": 0.78
        }
    ]
}
```

---

## Kullanım Örnekleri

### Örnek 1: Türkçe App Store Yorumları

```python
# Türkçe app store yorumları için konfigürasyon
seçenekler = {
    "LEMMATIZE": False,                    # Türkçe için gerekli değil
    "N_TOPICS": 20,                       # Konu başına 20 kelime
    "DESIRED_TOPIC_COUNT": 6,             # 6 konu çıkar
    "tokenizer_type": "bpe",              # BPE tokenizasyon kullan
    "nmf_type": "nmf",                    # Standart NMF
    "LANGUAGE": "TR",                     # Türkçe dil
    "separator": ",",                     # CSV ayırıcısı
    "gen_cloud": True,                    # Kelime bulutları oluştur
    "save_excel": True,                   # Excel'e dışa aktar
    "gen_topic_distribution": True        # Dağılım grafikleri oluştur
}

# Analizi çalıştır
sonuç = run_standalone_nmf(
    filepath="veri/uygulama_yorumlari.csv",
    table_name="uygulama_yorumlari_analizi",
    desired_columns="yorum_metni",
    options=seçenekler
)

print(f"Analiz tamamlandı: {sonuç['state']}")
print(f"Çıkarılan konular: {len(sonuç['topic_word_scores'])}")
```

### Örnek 2: Türkçe Akademik Makaleler

```python
# Türkçe araştırma makaleleri için konfigürasyon
seçenekler = {
    "LEMMATIZE": False,                    # Türkçe için lemmatizasyon yok
    "N_TOPICS": 25,                       # Konu başına 25 kelime
    "DESIRED_TOPIC_COUNT": 10,            # 10 konu çıkar
    "tokenizer_type": "bpe",              # BPE tokenizer
    "nmf_type": "opnmf",                  # Ortogonal NMF
    "LANGUAGE": "TR",                     # Türkçe dil
    "separator": ",",
    "gen_cloud": True,
    "save_excel": True,
    "gen_topic_distribution": True
}

# Analizi çalıştır
sonuç = run_standalone_nmf(
    filepath="veri/akademik_makaleler.csv",
    table_name="akademik_analiz",
    desired_columns="özet",
    options=seçenekler
)
```

### Örnek 3: Çoklu Dosya Toplu İşleme

```python
işlenecek_dosyalar = [
    {
        "filepath": "veri/veriset1.csv",
        "table_name": "analiz_1", 
        "column": "metin_içeriği",
        "konular": 8
    },
    {
        "filepath": "veri/veriset2.csv",
        "table_name": "analiz_2",
        "column": "açıklama",
        "konular": 12
    }
]

sonuçlar = []
for dosya_config in işlenecek_dosyalar:
    seçenekler = {
        "LEMMATIZE": False,
        "N_TOPICS": 15,
        "DESIRED_TOPIC_COUNT": dosya_config["konular"],
        "tokenizer_type": "bpe",
        "nmf_type": "nmf",
        "LANGUAGE": "TR",
        "separator": ",",
        "gen_cloud": True,
        "save_excel": True,
        "gen_topic_distribution": True
    }
    
    sonuç = run_standalone_nmf(
        filepath=dosya_config["filepath"],
        table_name=dosya_config["table_name"],
        desired_columns=dosya_config["column"],
        options=seçenekler
    )
    sonuçlar.append(sonuç)

print(f"{len(sonuçlar)} dosya başarıyla işlendi")
```

### Örnek 4: Sosyal Medya Analizi

```python
# Twitter/X gönderileri analizi için özelleştirilmiş konfigürasyon
seçenekler = {
    "LEMMATIZE": False,
    "N_TOPICS": 12,                       # Kısa metinler için daha az kelime
    "DESIRED_TOPIC_COUNT": 15,            # Çeşitlilik için daha fazla konu
    "tokenizer_type": "wordpiece",        # Sosyal medya için WordPiece
    "nmf_type": "opnmf",                  # Daha iyi ayrım için OPNMF
    "LANGUAGE": "TR",
    "separator": ",",
    "gen_cloud": True,
    "save_excel": True,
    "gen_topic_distribution": True
}

sonuç = run_standalone_nmf(
    filepath="veri/sosyal_medya_gönderileri.csv",
    table_name="sosyal_medya_analizi",
    desired_columns="gönderi_metni",
    options=seçenekler
)
```

---

## Sorun Giderme

### Yaygın Sorunlar ve Çözümler

#### 1. Dosya Kodlama Sorunları

**Sorun**: CSV dosyalarını okurken `UnicodeDecodeError`

**Çözüm**:
```python
# Sistem bunu otomatik olarak halleder, ancak sorun yaşarsanız:
# - CSV dosyanızın UTF-8 kodlamasında olduğundan emin olun
# - Verinizdeki özel karakterleri kontrol edin
# - Dosya kodlamasını UTF-8'e dönüştürmek için bir metin editörü kullanın
```

#### 2. Büyük Veri Setleriyle Bellek Sorunları

**Sorun**: `MemoryError` veya sistem yavaşlaması

**Çözüm**:
- Konu sayısını azaltın (`DESIRED_TOPIC_COUNT`)
- Veri setinizi daha küçük parçalara filtreleyin
- Konu başına daha az kelime kullanın (`N_TOPICS`)
- Daha güçlü bir makine kullanmayı düşünün

#### 3. Konu Üretilmemesi

**Sorun**: Boş veya anlamsız konular

**Çözümler**:
- Metin sütununuzun anlamlı içerik barındırıp barındırmadığını kontrol edin
- Minimum doküman uzunluğunu artırın
- `desired_topic_count` parametresini ayarlayın
- Diliniz için uygun metin temizleme yapıldığından emin olun

#### 4. Tokenizer Eğitimi Başarısızlığı

**Sorun**: Türkçe metin üzerinde tokenizer eğitimi başarısız oluyor

**Çözümler**:
- Yeterli metin verisi olduğundan emin olun (minimum 1000 doküman önerilir)
- Metin kalitesini kontrol edin ve aşırı gürültüyü kaldırın
- "bpe" ve "wordpiece" tokenizer'ları arasında geçiş yapmayı deneyin
- Metnin gerçekten Türkçe olduğunu doğrulayın

#### 5. Düşük Konu Kalitesi

**Sorun**: Konular çoğunlukla durak kelimeler veya alakasız terimler içeriyor

**Çözümler**:
- Metin ön işlemeyi geliştirin
- Alana özgü durak kelimeler ekleyin
- NMF parametrelerini ayarlayın (`norm_thresh`)
- Farklı `nmf_type` deneyin ("nmf" vs "opnmf")

#### 6. Veritabanı Kilit Hataları

**Sorun**: `database is locked` hatası

**Çözüm**:
```python
# Güvenliyse mevcut veritabanı dosyalarını silin:
# rm instance/topics.db
# rm instance/scopus.db
# Veya her çalıştırma için farklı table_name kullanın
```

#### 7. Eksik Bağımlılıklar

**Sorun**: Gerekli paketler için import hataları

**Çözüm**:
```bash
# uv kullanarak yükleyin (önerilen)
uv pip install -r requirements.txt

# Veya pip kullanarak
pip install -r requirements.txt
```

### Performans Optimizasyon İpuçları

1. **Büyük Veri Setleri İçin**:
   - Grafik oluşturmayı atlamak için `gen_topic_distribution=False` kullanın
   - Kelime bulutu oluşturmayı atlamak için `gen_cloud=False` ayarlayın
   - `N_TOPICS`'i 10-15'e düşürün

2. **Daha İyi Konu Kalitesi İçin**:
   - Daha ayrıntılı konular için `desired_topic_count`'u artırın
   - Daha iyi konu ayrımı için `nmf_type="opnmf"` kullanın
   - İngilizce metin için `LEMMATIZE=True` etkinleştirin

3. **Daha Hızlı İşleme İçin**:
   - Türkçe için `tokenizer_type="bpe"` kullanın (genellikle daha hızlı)
   - Çok kısa dokümanları kaldırmak için verinizi önceden filtreleyin
   - Daha iyi I/O performansı için SSD depolama kullanın

### Yardım Alma

Burada kapsanmayan sorunlarla karşılaşırsanız:

1. Ayrıntılı hata mesajları için konsol çıktısını kontrol edin
2. Giriş veri formatınızı ve içeriğinizi doğrulayın
3. Tüm bağımlılıkların düzgün şekilde yüklendiğinden emin olun
4. Kullanım durumunuz için konfigürasyon parametrelerini gözden geçirin

### Türkçe Dil Desteği İpuçları

1. **Türkçe Karakterler**: Sistem Türkçe karakterleri (ç, ğ, ı, ö, ş, ü) tam olarak destekler
2. **Tokenizasyon**: BPE genellikle Türkçe'nin ek yapısı için daha iyidir
3. **Metin Temizleme**: Türkçe özel karakterler ve noktalama işaretleri otomatik olarak işlenir
4. **Durak Kelimeler**: Sistem yaygın Türkçe durak kelimeleri otomatik olarak filtreler

### Veri Hazırlama Önerileri

1. **CSV Formatı**: UTF-8 kodlama kullanın
2. **Metin Sütunu**: Her satırda anlamlı metin olduğundan emin olun
3. **Veri Boyutu**: En az 100-200 doküman, ideal olarak 1000+ doküman
4. **Metin Uzunluğu**: Çok kısa (< 10 kelime) veya çok uzun (> 1000 kelime) metinlerden kaçının

---

*Bu dokümantasyon, tam NMF Standalone sistemini kapsamaktadır. Ek destek veya özellik istekleri için lütfen proje deposuna başvurun.*