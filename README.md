# Türkçe Tıbbi Metinlerden Öğrenen Alana Özgü Büyük Dil Modeli Geliştirilmesi

Bu proje, kranial difüzyon MR bulgularından Türkçe radyoloji raporu üretebilen alana özgü bir büyük dil modeli geliştirmeyi amaçlamaktadır. Çalışma kapsamında, yapılandırılmış klinik girdileri doğal, tutarlı ve klinik açıdan anlamlı raporlara dönüştürmek için farklı açık kaynaklı dil modelleri fine-tuning yöntemiyle eğitilmiş ve performansları karşılaştırılmıştır.

## Proje Kapsamı

Bu çalışmada farklı açık kaynaklı büyük dil modelleri aynı veri yapısı, benzer eğitim ayarları ve ortak değerlendirme metrikleriyle karşılaştırılmıştır.

Kullanılan temel yaklaşımlar:

- Instruction tuning formatında veri hazırlama
- Unsloth framework kullanımı
- 4-bit model yükleme
- QLoRA / LoRA adaptörleri
- Gradient checkpointing
- 8-bit optimizer
- Train / eval loss takibi
- ROUGE-L ve BERTScore metrikleriyle değerlendirme

## Klasör Yapısı

```text
STROKE_FINE_TUNING/
│
├── dataset/
│   ├── bitiyor_dataset.json
│   └── test_data.json
│
├── output/
│
├── SON.ipynb
├── SON.py
├── .env
├── README.md
├── Türkçe Radyoloji Raporu Üretimi_final.docx
├── Türkçe Radyoloji Raporu Üretimi_final.pdf
├── Türkçe Radyoloji Raporu Üretimi_final.pptx
└── .gitignore
```

## Kurulum

Gerekli Python ortamı oluşturulur ve  kütüphaneler yüklenir.

```bash
conda create -n stroke_fine_tuning python=3.10
conda activate stroke_fine_tuning
```


```bash
pip install torch transformers datasets accelerate evaluate rouge-score bert-score trl peft bitsandbytes
pip install unsloth
```

> Not: Kullanılan CUDA, PyTorch ve Unsloth sürümleri bilgisayarın GPU / CUDA uyumluluğuna göre değişebilir.

## Ortam Değişkenleri

Proje çalıştırılmadan önce ana dizinde `.env` dosyası oluşturulmalıdır.

Örnek `.env` yapısı:

```env
HF_TOKEN=your_huggingface_token_here

OUTPUT_DIR=C:\path\to\stroke_fine_tuning\output
JSON_PATH=C:\path\to\stroke_fine_tuning\dataset\bitiyor_dataset.json
TEST_DATA_PATH=C:\path\to\stroke_fine_tuning\dataset\test_data.json
BASE_MODEL_DIR=C:\path\to\huggingface\hub
```

## Kullanım

VSCode, Notebook veya Python dosyası üzerinden çalıştırılabilir.

Notebook için:

```bash
jupyter notebook SON.ipynb
```

Python dosyası için:

```bash
python SON.py
```

Eğitilecek model kod içerisinde seçilir. Örneğin:

```python
BASE_MODEL_KEY = "llama3b"
```

Bu seçimle ilgili model yüklenir, eğitim yapılır ve sonuçlar ilgili çıktı klasörüne kaydedilir.

## Değerlendirme Metrikleri

Projede model performansını değerlendirmek için aşağıdaki metrikler kullanılmıştır:

### 1. ROUGE-L

Üretilen rapor ile gerçek rapor arasındaki en uzun ortak alt diziyi dikkate alır. Metinsel benzerliği ölçmek için kullanılır.

### 2. BERTScore

Üretilen ve gerçek raporlar arasındaki anlamsal benzerliği ölçer. Kelime düzeyinden ziyade bağlamsal yakınlığa odaklanır.


## Kullanılan Modeller

Projede farklı model aileleri üzerinde denemeler yapılmıştır. Örnek modeller:

- Llama 3.2 1B Instruct
- Llama 3.2 3B Instruct
- Llama 3.1 / 3.x 8B tabanlı modeller
- Qwen 1.5B / 3B
- Phi-3 Mini
- Mistral 7B
- Medikal veya Türkçe odaklı varyantlar

## Notlar

- Büyük model dosyaları GitHub'a yüklenmemiştir.
- `.env` dosyası ve ara dosyalar `.gitignore` ile dışlanmıştır.
- Veri setinin paylaşılabilirliği etik ve gizlilik açısından ayrıca kontrol edilmelidir.

## Lisans

Bu proje bitirme projesi kapsamında geliştirilmiştir. Kullanılan modellerin ve veri setlerinin lisans koşulları ayrıca dikkate alınmalıdır.

## Contributors

Bu proje bitirme projesi kapsamında ekip çalışması olarak geliştirilmiştir.

- @helyaKi
- @zeynepzeren