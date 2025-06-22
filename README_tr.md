# NMF Standalone

Bu proje, Negatif Olmayan Matris Faktörizasyonu (NMF) kullanarak metin verileri üzerinde konu modelleme yapar. Hem İngilizce hem de Türkçe dilleri destekler ve `.csv` ile `.xlsx` dosyalarını işleyebilir. Ana betik olan `standalone_nmf.py`, veri ön işlemeden konu çıkarımı ve görselleştirmeye kadar tüm süreci yönetir.

## Proje Yapısı

```
nmf-standalone/
├── functions/
│   ├── english/
│   ├── nmf/
│   ├── tfidf/
│   └── turkish/
├── utils/
│   ├── other/
├── veri_setleri/
├── instance/
├── Output/
├── pyproject.toml
├── README.md
├── requirements.txt
├── standalone_nmf.py
└── uv.lock
```

-   **`functions/`**: NMF sürecinin temel mantığını içerir; İngilizce ve Türkçe metin işleme, TF-IDF hesaplama ve NMF algoritmaları için ayrı modüller bulunur.
-   **`utils/`**: Kelime bulutları oluşturma, tutarlılık skorları hesaplama ve sonuçları dışa aktarma gibi görevler için yardımcı fonksiyonları içerir.
-   **`veri_setleri/`**: Giriş veri setleri için varsayılan dizin.
-   **`instance/`**: İşlem sırasında oluşturulan veritabanlarını saklar (örn. `topics.db`, `scopus.db`).
-   **`Output/`**: Konu raporları, kelime bulutları ve dağılım grafikleri gibi tüm çıktı dosyalarının kaydedildiği dizin.
-   **`standalone_nmf.py`**: Konu modelleme sürecini çalıştırmak için ana yürütülebilir betik.
-   **`requirements.txt`**: Proje için gerekli Python paketlerinin listesi.

## Kurulum

Bu projeyi çalıştırmak için sanal ortam kullanılması önerilir.

1.  **Sanal ortam oluşturun:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **`uv` kullanarak bağımlılıkları yükleyin:**

    Bu proje hızlı bağımlılık yönetimi için `uv` kullanır. Eğer `uv`'niz yoksa, resmi talimatları takip ederek yükleyebilirsiniz.

    ```bash
    uv pip install -r requirements.txt
    ```

## Kullanım

Konu modellemesini çalıştırmak için ana giriş noktası `standalone_nmf.py` betiğidir. Analiziniz için parametreleri ayarlamak üzere bu betiği değiştirebilirsiniz.

Betik içinde yapılandırmanız ve çalıştırmanız gereken `run_standalone_nmf` fonksiyonudur.

Bu fonksiyonu betik içinde nasıl çağırabileceğinizin bir örneği:

```python
if __name__ == "__main__":
    # Türkçe veri seti örneği
    run_standalone_nmf(
        filepath="veri_setleri/turkce_veriniz.csv",
        table_name="turkce_analiz",
        desired_columns="metin_sutunu",
        desired_topic_count=10,
        LEMMATIZE=False,
        N_TOPICS=15,
        tokenizer_type="bpe",
        LANGUAGE="TR",
        nmf_type="nmf",
        separator=","
    )

    # İngilizce veri seti örneği
    run_standalone_nmf(
        filepath="veri_setleri/ingilizce_veriniz.csv",
        table_name="ingilizce_analiz",
        desired_columns="text_column",
        desired_topic_count=8,
        LEMMATIZE=True,
        N_TOPICS=20,
        tokenizer_type=None,
        LANGUAGE="EN",
        nmf_type="opnmf"
    )

```

Betiği çalıştırmak için terminalinizden basitçe şunu çalıştırın:

```bash
python standalone_nmf.py
```

### Parametreler

-   `filepath`: Giriş `.csv` veya `.xlsx` dosyanızın yolu.
-   `table_name`: Analiz çalışmanız için benzersiz bir ad. Bu, çıktı dosyalarını ve veritabanı tablolarını adlandırmak için kullanılır.
-   `desired_columns`: Veri dosyanızda analiz edilecek metni içeren sütunun adı.
-   `desired_topic_count`: Çıkarılacak konu sayısı.
-   `LEMMATIZE`: İngilizce metin için lemmatization'ı etkinleştirmek üzere `True` olarak ayarlayın.
-   `N_TOPICS`: Her konu için gösterilecek en önemli kelime sayısı.
-   `tokenizer_type`: Türkçe için `"bpe"` (Byte-Pair Encoding) veya `"wordpiece"` arasından seçebilirsiniz.
-   `LANGUAGE`: Türkçe için `"TR"` veya İngilizce için `"EN"`.
-   `nmf_type`: Kullanılacak NMF algoritması (`"nmf"` veya `"opnmf"`).
-   `separator`: `.csv` dosyanızda kullanılan ayırıcı (örn. `,`, `;`).

## Çıktılar

Betik, `table_name`'inizin adını taşıyan bir alt dizinde organize edilmiş olarak `Output/` dizininde çeşitli çıktılar oluşturur:

-   **Konu-Kelime Excel Dosyası**: Her konu için en önemli kelimeleri ve skorlarını içeren `.xlsx` dosyası.
-   **Kelime Bulutları**: Her konu için kelime bulutlarının PNG görüntüleri.
-   **Konu Dağılım Grafiği**: Dokümanların konulara göre dağılımını gösteren grafik.
-   **Tutarlılık Skorları**: Konular için tutarlılık skorlarını içeren JSON dosyası.
-   **En İyi Dokümanlar**: Her konu için en temsili dokümanları listeleyen JSON dosyası.