# %%
# ============================================================
# Hücre 1: Tüm Importlar ve Ortam Kontrolü
# Notebook boyunca gereken tüm kütüphaneler burada tek seferde yüklenir.
# Diğer hücrelerde ayrıca import yapılmasına gerek yoktur.
# ============================================================

import os


# PyTorch Inductor'ın C++ derleyicisi (cl) aramasını tamamen devre dışı bırakır
os.environ["TORCH_INDUCTOR_COMPILE_FX"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Alternatif olarak Inductor'a açıkça C++ optimizasyonlarını atlamasını söyleyelim
import torch
if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
    torch._inductor.config.cpp.vec_isa_ok = False
import gc
import json
import random
import re
import shutil
import unicodedata
from pathlib import Path

import torch
import unsloth
import numpy as np
import pandas as pd
import evaluate
import matplotlib.pyplot as plt

from tqdm import tqdm
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TextStreamer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

# .env kullanımı opsiyoneldir.
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    load_dotenv = None
    DOTENV_AVAILABLE = False

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("GPU: CUDA destekli GPU bulunamadı.")

print(f"Unsloth: {unsloth.__version__}")

# ============================================================
# Model seçimi
# ============================================================ 
# Her çalıştırmada SADECE bir model seçin.
# Geçerli seçenekler: "meditron7b", "gemma2b", "llama3b", "llama1b", "llama8b", "qwen1.5b", "qwen3b", "phi3mini", "mistral7b", "turkish-llama8b", "llamacare8b", "doktor-llama8b", "turkish-gemma9b", "doktor-llama8b-cosmos", "biomistral7b"

BASE_MODEL_KEY = "turkish-gemma9b"



# %%
colors  = ["#F86429", "#005c65", "#214c58", "#d9ec53", "#f5cd2e"]

# %% [markdown]
# gpu ve disk temizleme

# %%
# ============================================================
# Hücre 2: GPU / RAM ve Seçili Modelin Eski Çıktılarını Temizleme
# Her yeni model eğitiminden önce çalıştırılabilir.
#
# Not:
# - GPU belleğini ve büyük Python nesnelerini temizler.
# - .env içindeki OUTPUT_DIR ana klasörü altında BASE_MODEL_KEY klasörünü temizler.
# - Performans analizi için üretilmiş .png grafik dosyalarını korur.
# ============================================================

if DOTENV_AVAILABLE:
    load_dotenv()

#if "BASE_MODEL_KEY" not in globals():
    #raise RuntimeError("BASE_MODEL_KEY tanımlı değil. Hücre 1'de model seçimini yapın.")

# BASE_MODEL_KEY = "mistral7b"

VALID_MODEL_KEYS = ["meditron7b", "gemma2b", "llama3b", "llama1b", "llama8b", "qwen1.5b", "qwen3b", "phi3mini", "mistral7b", "turkish-llama8b", "llamacare8b", "doktor-llama8b", "turkish-gemma9b", "doktor-llama8b-cosmos", "biomistral7b"]
if BASE_MODEL_KEY not in VALID_MODEL_KEYS:
    raise ValueError(
        f"Geçersiz BASE_MODEL_KEY: {BASE_MODEL_KEY}. "
        f"Geçerli seçenekler: {VALID_MODEL_KEYS}"
    )


def get_required_env_path(env_key: str) -> Path:
    """.env içinden zorunlu bir path değerini okur ve Path olarak döndürür."""
    value = os.getenv(env_key)
    if value is None or str(value).strip() == "":
        raise RuntimeError(
            f".env içinde {env_key} tanımlı değil. "
            f"Lütfen .env dosyasına {env_key}=... satırını ekleyin."
        )
    return Path(value.strip().strip('"').strip("'"))


def show_gpu_memory(title="GPU bellek durumu"):
    print("=" * 60)
    print(title)
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA mevcut değil.")
        return

    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved  = torch.cuda.memory_reserved(device) / (1024 ** 3)
    free      = free_bytes / (1024 ** 3)
    total     = total_bytes / (1024 ** 3)

    print(f"GPU              : {torch.cuda.get_device_name(device)}")
    print(f"Allocated VRAM   : {allocated:.2f} GiB")
    print(f"Reserved VRAM    : {reserved:.2f} GiB")
    print(f"Free VRAM        : {free:.2f} GiB")
    print(f"Total VRAM       : {total:.2f} GiB")


def clean_model_output_dir_preserve_png(output_dir):
    """
    Verilen model çıktı klasöründe .png dosyalarını korur.
    .png dışındaki dosyaları ve boş kalan klasörleri siler.
    Böylece performans grafikleri kalır; eski model/checkpoint/log/prediction dosyaları temizlenir.
    """
    output_dir = Path(output_dir)

    print("=" * 60)
    print("Seçili model için eski çıktı klasörü temizliği")
    print("=" * 60)
    print(f"Seçili model : {BASE_MODEL_KEY}")
    print(f"Hedef klasör : {output_dir}")

    if not output_dir.exists():
        print("Hedef klasör bulunamadı; silinecek eski çıktı yok.")
        return

    deleted_files = 0
    deleted_dirs = 0
    kept_png = 0

    # Önce dosyaları temizle: PNG kalsın, diğer dosyalar silinsin.
    for path in sorted(output_dir.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if path.is_file():
            if path.suffix.lower() == ".png":
                kept_png += 1
                continue
            try:
                path.unlink()
                deleted_files += 1
            except PermissionError:
                print(f"Silinemedi, dosya kullanımda olabilir: {path}")

    # Sonra boş kalan klasörleri sil. PNG içeren klasörler boş kalmayacağı için korunur.
    for path in sorted(output_dir.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
                deleted_dirs += 1
            except OSError:
                pass

    print(f"Silinen dosya sayısı : {deleted_files}")
    print(f"Silinen klasör sayısı: {deleted_dirs}")
    print(f"Korunan PNG sayısı   : {kept_png}")


show_gpu_memory("Temizlik öncesi GPU bellek durumu")

# Önceki model/eğitimden kalabilecek büyük nesneleri sil.
for var_name in [
    "model",
    "tokenizer",
    "trainer",
    "trainer_stats",
    "train_dataset",
    "test_dataset",
    "dataset",
    "split_dataset",
    "raw_data",
    "outputs",
    "inputs",
    "pred_outputs",
    "true_outputs",
    "bert_results",
    "rouge_results",
]:
    if var_name in globals():
        del globals()[var_name]

# Python RAM temizliği
gc.collect()

# CUDA cache temizliği
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    torch.cuda.reset_peak_memory_stats()
    print("GPU belleği ve Python referansları temizlendi.")
else:
    print("CUDA mevcut değil; yalnızca Python RAM temizliği yapıldı.")

show_gpu_memory("Temizlik sonrası GPU bellek durumu")

# ------------------------------------------------------------
# Seçili modelin eski çıktı temizliği
# ------------------------------------------------------------
# .env içindeki OUTPUT_DIR ana çıktı klasörüdür.
# Örn: OUTPUT_DIR=C:\\...\\output_train ise temizlenecek klasör:
#      C:\\...\\output_train\\llama3b
CLEAN_BASE_OUTPUT_DIR = get_required_env_path("OUTPUT_DIR")
CLEAN_OUTPUT_DIR = CLEAN_BASE_OUTPUT_DIR / BASE_MODEL_KEY

clean_model_output_dir_preserve_png(CLEAN_OUTPUT_DIR)


# %% [markdown]
# **EĞİTİM AŞAMASI**

# %% [markdown]
# Yapılandırma ve Model Yükleme
# 

# %%
# ============================================================
# Hücre 3: Yapılandırma ve Model Yükleme
# Tek seferde tek model eğitimi + model adına özel çıktı klasörü
# ============================================================

# .env kullanıyorsanız yükle
if DOTENV_AVAILABLE:
    load_dotenv()
    print(".env dosyası yüklendi.")
else:
    print("python-dotenv kurulu değil. .env kullanılmayacak.")
    print("Gerekirse kurulum: pip install python-dotenv")


# ------------------------------------------------------------
# 1) Ana ayarlar
# ------------------------------------------------------------

# Model seçimi Hücre 1'de yapılır:
# BASE_MODEL_KEY = "llama3b"
# Geçerli seçenekler: "meditron7b", "gemma2b", "llama3b", "llama1b", "llama8b", "qwen1.5b", "qwen3b", "phi3mini", "mistral7b", "turkish-llama8b", "llamacare8b", "doktor-llama8b", "turkish-gemma9b", "doktor-llama8b-cosmos", "biomistral7b"
#if "BASE_MODEL_KEY" not in globals():
 #   BASE_MODEL_KEY = "llama3b"

VALID_MODEL_KEYS = ["meditron7b", "gemma2b", "llama3b", "llama1b", "llama8b", "qwen1.5b", "qwen3b", "phi3mini", "mistral7b", "turkish-llama8b", "llamacare8b", "doktor-llama8b", "turkish-gemma9b", "doktor-llama8b-cosmos", "biomistral7b"]
if BASE_MODEL_KEY not in VALID_MODEL_KEYS:
    raise ValueError(
        f"Geçersiz BASE_MODEL_KEY: {BASE_MODEL_KEY}. "
        f"Geçerli seçenekler: {VALID_MODEL_KEYS}"
    )


def get_required_env_value(env_key: str) -> str:
    """.env içinden zorunlu bir değeri okur."""
    value = os.getenv(env_key)
    if value is None or str(value).strip() == "":
        raise RuntimeError(
            f".env içinde {env_key} tanımlı değil. "
            f"Lütfen .env dosyasına {env_key}=... satırını ekleyin."
        )
    return str(value).strip().strip('"').strip("'")


def get_required_env_path(env_key: str) -> Path:
    """.env içinden zorunlu bir path değerini okur ve Path olarak döndürür."""
    return Path(get_required_env_value(env_key))


# .env dosyasındaki path'ler
# OUTPUT_DIR      : bütün model çıktılarının ana klasörü
# JSON_PATH       : eğitim/test dataset dosyası
# BASE_MODEL_DIR  : HuggingFace hub/cache ana klasörü
BASE_OUTPUT_DIR = get_required_env_path("OUTPUT_DIR")
JSON_PATH = get_required_env_path("JSON_PATH")
BASE_MODEL_DIR = get_required_env_path("BASE_MODEL_DIR")
HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN is not None:
    HF_TOKEN = str(HF_TOKEN).strip().strip('"').strip("'")


# ------------------------------------------------------------
# 2) Model klasör eşleştirmeleri
# ------------------------------------------------------------
# Sol taraf: notebook içinde seçeceğiniz BASE_MODEL_KEY.
# Sağ taraf: HuggingFace cache/hub altında beklenen model klasörü.
# Kendi bilgisayarınızdaki klasör adı farklıysa sadece sağ tarafı değiştirin.

MODEL_DIRS = {
    "meditron7b": "models--epfl-llm--meditron-7b",
    "gemma2b":  "models--unsloth--gemma-2-2b-it-bnb-4bit",
    "llama1b":  "models--unsloth--Llama-3.2-1B-Instruct-bnb-4bit",
    "llama3b":  "models--unsloth--Llama-3.2-3B-Instruct-bnb-4bit",
    "llama8b":  "models--unsloth--llama-3-8b-Instruct-bnb-4bit",
    "qwen1.5b": "models--unsloth--Qwen2.5-1.5B-Instruct-bnb-4bit",
    "qwen3b":   "models--unsloth--Qwen2.5-3B-Instruct-bnb-4bit",
    "phi3mini": "models--unsloth--Phi-3-mini-4k-instruct-bnb-4bit",
    "mistral7b": "models--unsloth--mistral-7b-instruct-v0.3-bnb-4bit",
    "biomistral7b": "models--BioMistral--BioMistral-7B",
    "turkish-llama8b": "models--ytu-ce-cosmos--Turkish-Llama-8b-v0.1",
    "llamacare8b": "models--Stephen-SMJ--LlamaCare",
    "doktor-llama8b": "models--alibayram--Doktor-Llama-3-8b",
    "turkish-gemma9b": "models--ytu-ce-cosmos--Turkish-Gemma-9b-T1",
    "doktor-llama8b-cosmos": "models--kayrab--doktor-llama-3-cosmos-8b-lora",
    
}

MODEL_BASE_DIR = BASE_MODEL_DIR / MODEL_DIRS[BASE_MODEL_KEY]

# Bu notebook boyunca tüm kayıtlar bu klasöre yapılır.
# Örn: OUTPUT_DIR ana klasörü output_train ise:
#      output_train\llama3b
OUTPUT_DIR = BASE_OUTPUT_DIR / BASE_MODEL_KEY
MODEL_OUTPUT_DIR = OUTPUT_DIR


# ------------------------------------------------------------
# 3) Yol kontrolleri
# ------------------------------------------------------------

if not BASE_MODEL_DIR.exists():
    raise FileNotFoundError(f"HuggingFace hub/cache ana klasörü bulunamadı: {BASE_MODEL_DIR}")

if not MODEL_BASE_DIR.exists():
    raise FileNotFoundError(
        f"Model klasörü bulunamadı: {MODEL_BASE_DIR}\n"
        f"BASE_MODEL_KEY doğruysa MODEL_DIRS içindeki klasör adını "
        f"kendi bilgisayarınızdaki adla eşleştirin."
    )

if not JSON_PATH.exists():
    raise FileNotFoundError(f"Dataset dosyası bulunamadı: {JSON_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 4) Snapshot klasörünü güvenli bul
# ------------------------------------------------------------

def find_snapshot_dir(model_base_dir: Path) -> Path:
    """
    HuggingFace cache yapısında snapshot klasörünü bulur.
    Şu iki durumu destekler:
    1) model_base_dir = models--... klasörü
    2) model_base_dir = doğrudan snapshot klasörü
    """

    # Eğer doğrudan model dosyalarının olduğu klasör verilmişse
    if (model_base_dir / "config.json").exists():
        return model_base_dir

    snapshots_dir = model_base_dir / "snapshots"

    if not snapshots_dir.exists():
        raise FileNotFoundError(
            f"'snapshots' klasörü bulunamadı: {snapshots_dir}\n"
            f"Eğer doğrudan snapshot klasörü kullanıyorsanız config.json içeren klasörü verin."
        )

    snapshot_paths = [
        p for p in snapshots_dir.iterdir()
        if p.is_dir() and (p / "config.json").exists()
    ]

    if not snapshot_paths:
        raise FileNotFoundError(
            f"{snapshots_dir} içinde config.json içeren snapshot klasörü bulunamadı."
        )

    # En son değiştirilen snapshot'ı seç
    snapshot_paths = sorted(snapshot_paths, key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshot_paths[0]


MODEL_NAME = find_snapshot_dir(MODEL_BASE_DIR)


# ------------------------------------------------------------
# 5) Bilgi yazdır ve seçim bilgisini kaydet
# ------------------------------------------------------------

run_info = {
    "base_model_key": BASE_MODEL_KEY,
    "model_cache_dir": str(MODEL_BASE_DIR),
    "selected_snapshot": str(MODEL_NAME),
    "dataset_path": str(JSON_PATH),
    "base_output_dir": str(BASE_OUTPUT_DIR),
    "model_output_dir": str(OUTPUT_DIR),
}

with open(OUTPUT_DIR / "selected_model_info.json", "w", encoding="utf-8") as f:
    json.dump(run_info, f, ensure_ascii=False, indent=2)

print("=" * 70)
print("AKTİF MODEL BİLGİSİ")
print("=" * 70)
print(f"Model anahtarı       : {BASE_MODEL_KEY}")
print(f"Model cache klasörü  : {MODEL_BASE_DIR}")
print(f"Kullanılan snapshot  : {MODEL_NAME}")
print(f"Dataset              : {JSON_PATH}")
print(f"Output klasörü       : {OUTPUT_DIR}")
print(f"CUDA available       : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU                  : {torch.cuda.get_device_name(0)}")
    print(f"Toplam VRAM          : {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GiB")

print("=" * 70)


# ------------------------------------------------------------
# 6) Model ve tokenizer yükleme
# ------------------------------------------------------------

print("Model ve tokenizer yükleniyor...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name       = str(MODEL_NAME),
    max_seq_length   = 2048,
    dtype            = None,
    load_in_4bit     = True,
    token            = HF_TOKEN,
    local_files_only = True,
)

# Bazı modellerde pad_token boş olabilir. Trainer ve generate için gerekli.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

print("Model ve tokenizer başarıyla yüklendi.")
print(f"Bu çalışmanın tüm çıktıları burada tutulacak: {OUTPUT_DIR}")


# %% [markdown]
# LoRA ayarı

# %%
# ============================================================
# Hücre 4: LoRA Ayarı
# 1B–3B instruct modeller için QLoRA ayarı
# ============================================================


# Önceki hücrede model yüklenmiş mi kontrol et
if "model" not in globals():
    raise RuntimeError("Önce model yükleme hücresini çalıştırmalısınız. 'model' değişkeni bulunamadı.")

if "tokenizer" not in globals():
    raise RuntimeError("Önce model yükleme hücresini çalıştırmalısınız. 'tokenizer' değişkeni bulunamadı.")


# ------------------------------------------------------------
# 16 GB VRAM için önerilen başlangıç ayarı
# ------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32

# Eğer CUDA out of memory alırsanız şunları kullanın:
# LORA_R = 8
# LORA_ALPHA = 16


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


print("=" * 70)
print("LoRA ayarı uygulanıyor...")
print(f"LoRA rank        : {LORA_R}")
print(f"LoRA alpha       : {LORA_ALPHA}")
print(f"Target modules   : {TARGET_MODULES}")
print("=" * 70)


model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    target_modules=TARGET_MODULES,
    bias="none",
    use_gradient_checkpointing="unsloth",
    use_rslora=True,
    random_state=3407,
)


print("LoRA başarıyla uygulandı.")

if torch.cuda.is_available():
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"Boş VRAM   : {free_bytes / (1024 ** 3):.2f} GiB")
    print(f"Toplam VRAM: {total_bytes / (1024 ** 3):.2f} GiB")



# %% [markdown]
# veriseti hazırlığı

# %%
# ============================================================
# Hücre 5: Veriseti Hazırlığı - Bellek Dostu Versiyon
# Eğitim dataset'i sadece "text" içerir.
# Test için input/output ayrı Python listesinde saklanır.
# ============================================================


if "JSON_PATH" not in globals():
    raise RuntimeError("JSON_PATH tanımlı değil. Önce yapılandırma hücresini çalıştırın.")

if "tokenizer" not in globals():
    raise RuntimeError("tokenizer tanımlı değil. Önce model yükleme hücresini çalıştırın.")


# ------------------------------------------------------------
# 1) JSON dosyasını oku
# ------------------------------------------------------------

with open(JSON_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

if not isinstance(raw_data, list):
    raise ValueError("JSON dosyası liste formatında olmalı.")

for i, ex in enumerate(raw_data):
    if "input" not in ex or "output" not in ex:
        raise ValueError(f"{i}. örnekte input veya output alanı eksik.")


# ------------------------------------------------------------
# 2) Train-test bölmesini raw_data üzerinde yap
# ------------------------------------------------------------

random.seed(3407)

indices = list(range(len(raw_data)))
random.shuffle(indices)

test_size = 0.2
test_count = int(len(indices) * test_size)

test_indices = indices[:test_count]
train_indices = indices[test_count:]

train_examples = [raw_data[i] for i in train_indices]
test_examples = [raw_data[i] for i in test_indices]


# ------------------------------------------------------------
# 3) Eğitim metni oluştur
# ------------------------------------------------------------

def make_training_text(example):
    input_json = json.dumps(example["input"], ensure_ascii=False)

    text = (
        "Aşağıdaki kranial difüzyon MR verilerine göre Türkçe radyoloji raporu üret.\n\n"
        "Girdi JSON:\n"
        f"{input_json}\n\n"
        "Rapor:\n"
        f"{example['output'].strip()}"
    )

    if tokenizer.eos_token is not None:
        text += tokenizer.eos_token

    return text


train_text_data = [
    {"text": make_training_text(ex)}
    for ex in train_examples
]

test_text_data = [
    {"text": make_training_text(ex)}
    for ex in test_examples
]

with open("C:\\Users\\merve\\Desktop\\Merve\\stroke_fine_tuning\\dataset\\test_data.json", "w", encoding="utf-8") as f:
    json.dump(test_text_data, f)
# ------------------------------------------------------------
# 4) Hugging Face Dataset oluştur
# ------------------------------------------------------------

train_dataset = Dataset.from_list(train_text_data)
test_dataset = Dataset.from_list(test_text_data)


print("Veri hazırlığı tamamlandı.")
print(f"Toplam veri      : {len(raw_data)}")
print(f"Eğitim örneği    : {len(train_dataset)}")
print(f"Test örneği      : {len(test_dataset)}")
print(f"Test raw örneği  : {len(test_examples)}")
print(f"Train kolonları  : {train_dataset.column_names}")
print(f"Test kolonları   : {test_dataset.column_names}")

print("\nÖrnek eğitim metni:")
print("-" * 70)
print(train_dataset[0]["text"][:1000])
print("-" * 70)



# %% [markdown]
# eğitimi başlat ve modeli kaydet

# %%
# ============================================================
# Hücre 6: Trainer Ayarları, Eğitim ve Model Kaydetme
# Bellek dostu text-only dataset formatına uygundur.
# Tüm çıktılar OUTPUT_DIR = output\<BASE_MODEL_KEY> altına kaydedilir.
# ============================================================


# ------------------------------------------------------------
# 1) Gerekli değişken kontrolleri
# ------------------------------------------------------------

#FastLanguageModel.for_training(model)

required_vars = [
    "model",
    "tokenizer",
    "train_dataset",
    "test_dataset",
    "OUTPUT_DIR",
    "BASE_MODEL_KEY",
]

for var in required_vars:
    if var not in globals():
        raise RuntimeError(f"'{var}' tanımlı değil. Önceki hücreleri sırayla çalıştırın.")

# ------------------------------------------------------------
# 2) Tokenizer güvenlik ayarları
# ------------------------------------------------------------

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# OUTPUT_DIR Path ise string'e çevir
OUTPUT_DIR = str(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 3) 16 GB VRAM için güvenli eğitim ayarları
# ------------------------------------------------------------

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 2048

# eval_steps=1 yapma; çok yavaşlatır.
EVAL_STEPS = 10
LOGGING_STEPS = 1

USE_BF16 = is_bfloat16_supported()
USE_FP16 = not USE_BF16

print("=" * 70)
print("EĞİTİM AYARLARI")
print("=" * 70)
print(f"Aktif model                 : {BASE_MODEL_KEY}")
print(f"Output klasörü             : {OUTPUT_DIR}")
print(f"Train örnek sayısı          : {len(train_dataset)}")
print(f"Test örnek sayısı           : {len(test_dataset)}")
print(f"Train kolonları             : {train_dataset.column_names}")
print(f"Test kolonları              : {test_dataset.column_names}")
print(f"Batch size                  : {PER_DEVICE_TRAIN_BATCH_SIZE}")
print(f"Gradient accumulation       : {GRADIENT_ACCUMULATION_STEPS}")
print(f"Efektif batch size          : {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Epoch                       : {NUM_TRAIN_EPOCHS}")
print(f"Learning rate               : {LEARNING_RATE}")
print(f"Max sequence length         : {MAX_SEQ_LENGTH}")
print(f"bf16                        : {USE_BF16}")
print(f"fp16                        : {USE_FP16}")

if torch.cuda.is_available():
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"Boş VRAM                    : {free_bytes / (1024 ** 3):.2f} GiB")
    print(f"Toplam VRAM                 : {total_bytes / (1024 ** 3):.2f} GiB")

print("=" * 70)

training_config = {
    "base_model_key": BASE_MODEL_KEY,
    "output_dir": OUTPUT_DIR,
    "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
    "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "max_seq_length": MAX_SEQ_LENGTH,
    "eval_steps": EVAL_STEPS,
    "logging_steps": LOGGING_STEPS,
    "bf16": USE_BF16,
    "fp16": USE_FP16,
}

with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w", encoding="utf-8") as f:
    json.dump(training_config, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------
# 4) TrainingArguments
# ------------------------------------------------------------
# Not:
# Bazı transformers sürümlerinde parametre adı eval_strategy,
# bazılarında evaluation_strategy olabilir.
# Önce eval_strategy ile deniyoruz.

try:
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=-1,

        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        gradient_checkpointing=True,
        optim="adamw_8bit",

        learning_rate=LEARNING_RATE,
        weight_decay=0.001,
        lr_scheduler_type="linear",
        warmup_steps=5,

        logging_steps=LOGGING_STEPS,

        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        save_strategy="epoch",
        save_total_limit=1,

        fp16=USE_FP16,
        bf16=USE_BF16,

        seed=3407,
        report_to="none",
        torch_compile=False,

        remove_unused_columns=False,
    )

except TypeError:
    print("eval_strategy parametresi desteklenmedi. evaluation_strategy ile tekrar deneniyor.")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=-1,

        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        gradient_checkpointing=True,
        optim="adamw_8bit",

        learning_rate=LEARNING_RATE,
        weight_decay=0.001,
        lr_scheduler_type="linear",
        warmup_steps=5,

        logging_steps=LOGGING_STEPS,

        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,

        save_strategy="epoch",
        save_total_limit=1,

        fp16=USE_FP16,
        bf16=USE_BF16,

        seed=3407,
        report_to="none",
        torch_compile=False,

        remove_unused_columns=False,
    )

# ------------------------------------------------------------
# 5) Trainer oluştur
# ------------------------------------------------------------

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,

    train_dataset=train_dataset,
    eval_dataset=test_dataset,

    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=1,
    packing=False,

    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
    ),

    args=args,
)

# ------------------------------------------------------------
# 6) Eğitim
# ------------------------------------------------------------

print("Eğitim başlıyor...")

try:
    trainer_stats = trainer.train()

    print("\nEğitim tamamlandı.")

    train_loss = trainer_stats.metrics.get("train_loss", None)

    if train_loss is not None:
        print(f"Train loss: {train_loss:.4f}")
    else:
        print("Train loss bilgisi bulunamadı.")

except torch.cuda.OutOfMemoryError:
    print("\nCUDA out of memory hatası alındı.")
    print("Önerilen çözümler:")
    print("1) Hücre 4'te LORA_R = 8, LORA_ALPHA = 16 yap.")
    print("2) Hücre 3'te max_seq_length = 1024 yap.")
    print("3) Hücre 2 GPU temizleme hücresini çalıştır.")
    print("4) Gerekirse VS Code kernel'i yeniden başlat.")
    raise

# ------------------------------------------------------------
# 7) LoRA adaptörü ve tokenizer kaydet
# ------------------------------------------------------------
# model.save_pretrained(OUTPUT_DIR) ile adapter dosyaları doğrudan
# output\<BASE_MODEL_KEY> içine kaydedilir.

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nLoRA adaptörü, tokenizer ve eğitim ayarları kaydedildi:")
print(OUTPUT_DIR)
print("Bu klasörde adapter_model.safetensors / adapter_config.json, checkpoint, log ve sonraki test dosyaları tutulacak.")



# %% [markdown]
# **TEST AŞAMASI**

# %% [markdown]
# train loss vs eval loss grafiği

# %%
# ============================================================
# Hücre 7: Eğitim Loglarını Kaydetme ve Loss Grafiği
# ============================================================


if "trainer" not in globals():
    raise RuntimeError("trainer tanımlı değil. Önce eğitim hücresini çalıştırın.")

if "OUTPUT_DIR" not in globals():
    raise RuntimeError("OUTPUT_DIR tanımlı değil.")

OUTPUT_DIR = str(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1) Trainer log geçmişini al
# ------------------------------------------------------------

log_history = trainer.state.log_history

if not log_history:
    raise RuntimeError("trainer.state.log_history boş. Eğitim hücresi tamamlanmamış olabilir.")

# ------------------------------------------------------------
# 2) Logları JSON olarak kaydet
# ------------------------------------------------------------

log_path = os.path.join(OUTPUT_DIR, "log_history.json")

with open(log_path, "w", encoding="utf-8") as f:
    json.dump(log_history, f, ensure_ascii=False, indent=2)

print("Log history kaydedildi:")
print(log_path)

# ------------------------------------------------------------
# 3) Train loss ve eval loss değerlerini ayıkla
# ------------------------------------------------------------

train_steps = []
train_loss = []

eval_steps = []
eval_loss = []

for log in log_history:
    if "loss" in log and "step" in log:
        train_steps.append(log["step"])
        train_loss.append(log["loss"])

    if "eval_loss" in log and "step" in log:
        eval_steps.append(log["step"])
        eval_loss.append(log["eval_loss"])

print(f"Train loss kayıt sayısı: {len(train_loss)}")
print(f"Eval loss kayıt sayısı : {len(eval_loss)}")

if not train_loss:
    raise RuntimeError("Train loss bulunamadı. Eğitim logları beklenen formatta değil.")

# ------------------------------------------------------------
# 4) Loss grafiği çiz
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    train_steps,
    train_loss,
    label="Train Loss",
    linewidth=2,
    marker="o",
    markersize=4,
    color=colors[0],
    markerfacecolor=colors[0],
    markeredgecolor=colors[0],
)

if eval_loss:
    plt.plot(
        eval_steps,
        eval_loss,
        label="Eval Loss",
        linewidth=2,
        marker="s",
        markersize=5,
        linestyle="--",
        color=colors[2],
        markerfacecolor=colors[2],
        markeredgecolor=colors[2],
    )
else:
    print("Eval loss bulunamadı. Grafik sadece train loss ile çizilecek.")

plt.title("Model Eğitim Grafiği: Train Loss ve Eval Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

graph_path = os.path.join(OUTPUT_DIR, "train_eval_loss.png")
plt.savefig(graph_path, dpi=150)
plt.show()

print("Loss grafiği kaydedildi:")
print(graph_path)



# %% [markdown]
# inference hazırlığı ve yanıt üretimi

# %%
# ============================================================
# Hücre 8: Değerlendirme Hazırlığı ve Yanıt Üretimi (Inference)
# test_examples kullanır
# ============================================================


required_vars = ["model", "tokenizer", "test_examples", "OUTPUT_DIR"]
for var in required_vars:
    if var not in globals():
        raise RuntimeError(f"'{var}' tanımlı değil. Önceki hücreleri sırayla çalıştırın.")

OUTPUT_DIR = str(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1) Eğitimde kullandığın prompt yapısına uyumlu inference promptu
# ------------------------------------------------------------

def build_inference_prompt(input_data):
    return (
        "Aşağıdaki kranial difüzyon MR verilerine göre Türkçe radyoloji raporu üret.\n\n"
        "Girdi JSON:\n"
        f"{json.dumps(input_data, ensure_ascii=False)}\n\n"
        "Rapor:\n"
    )

# ------------------------------------------------------------
# 2) Inference moduna geçir
# ------------------------------------------------------------

FastLanguageModel.for_inference(model)

true_outputs = []
pred_outputs = []
eval_inputs = []

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Test örnekleri için model yanıtları üretiliyor...")

for item in tqdm(test_examples, desc="Inference"):
    input_data = item["input"]
    gercek_rapor = item["output"]

    tokenizer.padding_side = "left"  # Tahmin (Inference) için ŞARTTIR!

    prompt = build_inference_prompt(input_data)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    ).to(device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        max_new_tokens=384,
        min_new_tokens=64,
        temperature=1.1,             # değerlendirme için deterministik olsun
        do_sample=False,              # değerlendirme için deterministik olsun
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    eval_inputs.append(input_data)
    true_outputs.append(gercek_rapor)
    pred_outputs.append(generated_text)

print("\nYanıt üretimi tamamlandı.")
print(f"Toplam test örneği: {len(pred_outputs)}")

# ------------------------------------------------------------
# 3) İlk birkaç örneği kontrol amaçlı göster
# ------------------------------------------------------------

preview_count = min(3, len(pred_outputs))

for i in range(preview_count):
    print("\n" + "=" * 80)
    print(f"ÖRNEK {i+1}")
    print("=" * 80)
    print("INPUT:")
    print(json.dumps(eval_inputs[i], ensure_ascii=False, indent=2))
    print("\nGERÇEK RAPOR:")
    print(true_outputs[i][:1000])
    print("\nMODEL RAPORU:")
    print(pred_outputs[i][:1000])

# ------------------------------------------------------------
# 4) Tahminleri kaydet
# ------------------------------------------------------------

predictions_df = pd.DataFrame({
    "input_json": [json.dumps(x, ensure_ascii=False) for x in eval_inputs],
    "true_output": true_outputs,
    "pred_output": pred_outputs,
})

predictions_csv_path = os.path.join(OUTPUT_DIR, "predictions.csv")
predictions_df.to_csv(predictions_csv_path, index=False, encoding="utf-8-sig")

print(f"\nTahminler kaydedildi: {predictions_csv_path}")



# %% [markdown]
# metriklerin hesaplanması ve sonuçlar

# %%
# ============================================================
# Hücre 9: ROUGE-L, BERTScore, Med-F1 Hesaplama ve Grafikler
# ============================================================


# ------------------------------------------------------------
# 0) Grafik renk paleti
# ------------------------------------------------------------

required_vars = ["pred_outputs", "true_outputs", "test_examples", "OUTPUT_DIR"]
for var in required_vars:
    if var not in globals():
        raise RuntimeError(f"'{var}' tanımlı değil. Önceki hücreleri sırayla çalıştırın.")

OUTPUT_DIR = str(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1) ROUGE-L ve BERTScore
# ------------------------------------------------------------

print("Metrikler hesaplanıyor...")

rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")

rouge_results = rouge_metric.compute(
    predictions=pred_outputs,
    references=true_outputs,
    rouge_types=["rougeL"]
)

overall_rouge_l = rouge_results["rougeL"]

per_sample_rouge_l = []
for pred, true in zip(pred_outputs, true_outputs):
    score = rouge_metric.compute(
        predictions=[pred],
        references=[true],
        rouge_types=["rougeL"]
    )["rougeL"]
    per_sample_rouge_l.append(score)

bert_results = bertscore_metric.compute(
    predictions=pred_outputs,
    references=true_outputs,
    lang="tr",
    model_type="dbmdz/bert-base-turkish-cased"
)

per_sample_bert_f1 = np.array(bert_results["f1"], dtype=float)
overall_bert_f1 = float(np.mean(per_sample_bert_f1))

# ------------------------------------------------------------
# 2) Med-F1 / Clinical Entity F1 için yardımcı fonksiyonlar
# ------------------------------------------------------------

def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower()

    tr_map = str.maketrans("çğıöşüâîû", "cgiosuaiu")
    text = text.translate(tr_map)

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def contains_any(text, patterns):
    return any(p in text for p in patterns)

# ------------------------------------------------------------
# 3) Klinik sözlükler / alias yapıları
# ------------------------------------------------------------

LOB_MAP = {
    "frontal": ["frontal", "fronto"],
    "parietal": ["parietal", "parieto"],
    "temporal": ["temporal", "temporo"],
    "oksipital": ["oksipital", "occipital", "oksipito", "occipito"],
    "insula": ["insula", "insular", "insuler"],
    "singulat": ["singulat", "cingulat"],
    "bazal gangliyon": ["bazal gangliyon", "basal gangliyon"],
    "serebellum": ["serebellum"],
    "pons": ["pons"],
    "talamus": ["talamus"],
    "paraventrikuler": ["paraventrikuler"],
    "sentrium": ["sentrium", "centrum"],
}

DERIN_MAP = {
    "bazal gangliyon": ["bazal gangliyon", "basal gangliyon"],
    "internal kapsul": ["internal kapsul", "ic kapsul", "i c kapsul"],
    "talamus": ["talamus"],
    "lentiform nukleus": ["lentiform nukleus"],
    "kaudat nukleus": ["kaudat nukleus", "nukleus kaudatus", "nucleus caudatus"],
    "putamen": ["putamen"],
    "pons": ["pons"],
    "serebellum": ["serebellum"],
    "medulla": ["medulla"],
    "mezensefalon": ["mezensefalon"],
    "hipokampus": ["hipokampus"],
    "sentrum semiovale": ["sentrum semiovale", "centrum semiovale"],
    "korona radiata": ["korona radiata"],
    "eksternal kapsul": ["eksternal kapsul"],
    "lateral ventrikul": ["lateral ventrikul"],
    "insula": ["insula", "insular", "insuler"],
}

DAMAR_MAP = {
    "MCA": ["mca"],
    "ACA": ["aca"],
    "PCA": ["pca"],
    "PICA": ["pica"],
    "SCA": ["sca"],
    "AICA": ["aica"],
    "lakuner": ["lakuner", "lakun"],
    "multiple": ["multiple", "multipl"],
}


def canonicalize_from_map(value, alias_map):
    value_norm = normalize_text(value)

    if not value_norm:
        return None

    for canonical, aliases in alias_map.items():
        for alias in aliases:
            if alias in value_norm:
                return canonical

    return value_norm

# ------------------------------------------------------------
# 4) Ground truth label set üretimi
# ------------------------------------------------------------

def build_gt_labels(input_data):
    labels = set()

    lat = input_data.get("lateralizasyon_gt")
    if lat is not None:
        lat_norm = normalize_text(lat)
        if lat_norm:
            labels.add(f"lat:{lat_norm}")

    for val in ensure_list(input_data.get("lob_gt")):
        canon = canonicalize_from_map(val, LOB_MAP)
        if canon:
            labels.add(f"lob:{canon}")

    for val in ensure_list(input_data.get("derin_gt")):
        canon = canonicalize_from_map(val, DERIN_MAP)
        if canon:
            labels.add(f"derin:{canon}")

    damar = input_data.get("damar_alani_gt")
    if damar is not None:
        canon = canonicalize_from_map(damar, DAMAR_MAP)
        if canon:
            labels.add(f"damar:{canon}")

    donem = input_data.get("donem_gt")
    if donem is not None:
        d = normalize_text(donem)

        if "akut-subakut" in d or "akut subakut" in d:
            labels.add("donem:akut-subakut")
        elif "subakut-kronik" in d or "subakut kronik" in d:
            labels.add("donem:subakut-kronik")
        elif "subakut" in d:
            labels.add("donem:subakut")
        elif "kronik" in d:
            labels.add("donem:kronik")
        elif "erken" in d:
            labels.add("donem:erken")
        elif "akut" in d:
            labels.add("donem:akut")

    labels.add(f"multifokal:{bool(input_data.get('multifokal_gt', False))}")
    labels.add(f"hemoraji:{bool(input_data.get('hemoraji_gt', False))}")
    labels.add(f"vazojenik_odem:{bool(input_data.get('vazojenik_odem_gt', False))}")
    labels.add(f"artefakt:{bool(input_data.get('artefakt_gt', False))}")

    return labels

# ------------------------------------------------------------
# 5) Tahmin edilen rapordan label set çıkarımı
# ------------------------------------------------------------

def extract_pred_labels(report_text):
    text = normalize_text(report_text)
    text_pad = f" {text} "

    labels = set()

    has_sag = (
        " sag " in text_pad
        or " sagda" in text_pad
        or " sag hemisfer" in text_pad
        or " sag tarafta" in text_pad
        or " sag front" in text_pad
        or " sag par" in text_pad
    )

    has_sol = (
        " sol " in text_pad
        or " solda" in text_pad
        or " sol hemisfer" in text_pad
        or " sol tarafta" in text_pad
        or " sol front" in text_pad
        or " sol par" in text_pad
    )

    if "bilateral" in text:
        labels.add("lat:bilateral")
    elif has_sag and has_sol:
        labels.add("lat:bilateral")
    elif has_sag:
        labels.add("lat:sag")
    elif has_sol:
        labels.add("lat:sol")

    for canonical, aliases in LOB_MAP.items():
        if contains_any(text, aliases):
            labels.add(f"lob:{canonical}")

    for canonical, aliases in DERIN_MAP.items():
        if contains_any(text, aliases):
            labels.add(f"derin:{canonical}")

    for canonical, aliases in DAMAR_MAP.items():
        if contains_any(text, aliases):
            labels.add(f"damar:{canonical}")

    if "akut subakut" in text or "akut-subakut" in text:
        labels.add("donem:akut-subakut")
    elif "subakut kronik" in text or "subakut-kronik" in text:
        labels.add("donem:subakut-kronik")
    elif "subakut" in text:
        labels.add("donem:subakut")
    elif "kronik" in text:
        labels.add("donem:kronik")
    elif "erken" in text:
        labels.add("donem:erken")
    elif "akut" in text or "akut iskemi" in text or "akut-iskemik" in text:
        labels.add("donem:akut")

    pred_multifokal = contains_any(
        text,
        ["multifokal", "birden fazla", "coklu odak", "daginik odak", "multipl"]
    )

    pred_hemoraji = contains_any(
        text,
        ["hemoraj", "hemorajik", "kanama", "hematom"]
    )

    pred_vazojenik_odem = contains_any(
        text,
        ["vazojenik odem", "vasojenik odem"]
    )

    pred_artefakt = contains_any(
        text,
        ["artefakt", "suboptimal", "hareket artefakti"]
    )

    labels.add(f"multifokal:{pred_multifokal}")
    labels.add(f"hemoraji:{pred_hemoraji}")
    labels.add(f"vazojenik_odem:{pred_vazojenik_odem}")
    labels.add(f"artefakt:{pred_artefakt}")

    return labels

# ------------------------------------------------------------
# 6) Med-F1 hesaplama
# ------------------------------------------------------------

def f1_from_sets(gt_set, pred_set):
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1, tp, fp, fn


per_sample_med_f1 = []
per_sample_med_precision = []
per_sample_med_recall = []

all_tp, all_fp, all_fn = 0, 0, 0

gt_label_sets = []
pred_label_sets = []

for example, pred_text in zip(test_examples, pred_outputs):
    gt_set = build_gt_labels(example["input"])
    pred_set = extract_pred_labels(pred_text)

    gt_label_sets.append(gt_set)
    pred_label_sets.append(pred_set)

    precision, recall, f1, tp, fp, fn = f1_from_sets(gt_set, pred_set)

    per_sample_med_precision.append(precision)
    per_sample_med_recall.append(recall)
    per_sample_med_f1.append(f1)

    all_tp += tp
    all_fp += fp
    all_fn += fn

overall_med_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
overall_med_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0

overall_med_f1 = (
    2 * overall_med_precision * overall_med_recall / (overall_med_precision + overall_med_recall)
    if (overall_med_precision + overall_med_recall) > 0
    else 0.0
)

# ------------------------------------------------------------
# 7) Sonuçları yazdır
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TEST PERFORMANS SONUÇLARI")
print("=" * 60)
print(f"ROUGE-L              : %{overall_rouge_l * 100:.2f}")
print(f"BERTScore F1         : %{overall_bert_f1 * 100:.2f}")
print(f"Med-F1               : %{overall_med_f1 * 100:.2f}")
print(f"Med-Precision        : %{overall_med_precision * 100:.2f}")
print(f"Med-Recall           : %{overall_med_recall * 100:.2f}")
print("=" * 60)

# ------------------------------------------------------------
# 8) Sonuçları dosyaya kaydet
# ------------------------------------------------------------

results_summary = {
    "rougeL": float(overall_rouge_l),
    "bertscore_f1": float(overall_bert_f1),
    "med_f1": float(overall_med_f1),
    "med_precision": float(overall_med_precision),
    "med_recall": float(overall_med_recall),
}

summary_json_path = os.path.join(OUTPUT_DIR, "metrics_summary.json")

with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

results_df = pd.DataFrame({
    "sample_id": list(range(len(pred_outputs))),
    "true_output": true_outputs,
    "pred_output": pred_outputs,
    "rougeL": per_sample_rouge_l,
    "bertscore_f1": per_sample_bert_f1,
    "med_f1": per_sample_med_f1,
    "med_precision": per_sample_med_precision,
    "med_recall": per_sample_med_recall,
    "gt_labels": [sorted(list(x)) for x in gt_label_sets],
    "pred_labels": [sorted(list(x)) for x in pred_label_sets],
})

results_csv_path = os.path.join(OUTPUT_DIR, "detailed_metrics.csv")
results_df.to_csv(results_csv_path, index=False, encoding="utf-8-sig")

print(f"Özet metrikler kaydedildi: {summary_json_path}")
print(f"Ayrıntılı sonuçlar kaydedildi: {results_csv_path}")

# ------------------------------------------------------------
# 9) Grafik 1: Genel metrik karşılaştırması
# ------------------------------------------------------------

metric_names = ["ROUGE-L", "BERTScore F1", "Med-F1"]
metric_values = [
    overall_rouge_l * 100,
    overall_bert_f1 * 100,
    overall_med_f1 * 100,
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    metric_names,
    metric_values,
    width=0.5,
    color=[colors[0], colors[2], colors[4]]
)

plt.ylim(0, 100)
plt.ylabel("Skor (%)")
plt.title("Genel Model Performansı")
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar, val in zip(bars, metric_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 1,
        f"%{val:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

overall_plot_path = os.path.join(OUTPUT_DIR, "overall_metrics_bar.png")
plt.savefig(overall_plot_path, dpi=150)
plt.show()

print(f"Genel metrik grafiği kaydedildi: {overall_plot_path}")

# ------------------------------------------------------------
# 10) Grafik 2: Örnek bazlı metrik dağılımı
# ------------------------------------------------------------

x = np.arange(len(pred_outputs))

plt.figure(figsize=(12, 6))

plt.plot(
    x,
    np.array(per_sample_rouge_l) * 100,
    label="ROUGE-L",
    linewidth=1.8,
    color=colors[0]
)

plt.plot(
    x,
    np.array(per_sample_bert_f1) * 100,
    label="BERTScore F1",
    linewidth=1.8,
    color=colors[2]
)

plt.plot(
    x,
    np.array(per_sample_med_f1) * 100,
    label="Med-F1",
    linewidth=1.8,
    color=colors[4]
)

plt.xlabel("Test Örneği")
plt.ylabel("Skor (%)")
plt.title("Örnek Bazlı Performans Dağılımı")
plt.ylim(0, 100)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

sample_plot_path = os.path.join(OUTPUT_DIR, "per_sample_metrics_line.png")
plt.savefig(sample_plot_path, dpi=150)
plt.show()

print(f"Örnek bazlı metrik grafiği kaydedildi: {sample_plot_path}")

# ------------------------------------------------------------
# 11) Grafik 3: Med-F1 detay grafiği
# ------------------------------------------------------------

med_metric_names = ["Med-Precision", "Med-Recall", "Med-F1"]
med_metric_values = [
    overall_med_precision * 100,
    overall_med_recall * 100,
    overall_med_f1 * 100,
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    med_metric_names,
    med_metric_values,
    width=0.5,
    color=[colors[1], colors[2], colors[4]]
)

plt.ylim(0, 100)
plt.ylabel("Skor (%)")
plt.title("Med-F1 Bileşenleri")
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar, val in zip(bars, med_metric_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 1,
        f"%{val:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

med_plot_path = os.path.join(OUTPUT_DIR, "med_f1_components_bar.png")
plt.savefig(med_plot_path, dpi=150)
plt.show()

print(f"Med-F1 bileşen grafiği kaydedildi: {med_plot_path}")




