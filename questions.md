<!-- @format -->

# Tugas Take-Home Sesi 05

## Advanced CNNs and Transfer Learning

**Topik:** Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

---

## Ringkasan Penugasan

| Informasi          | Keterangan                                                               |
| ------------------ | ------------------------------------------------------------------------ |
| Sifat tugas        | Individu, take-home, berbasis implementasi praktikum (_hands-on_)        |
| Bobot nilai        | 12,5% dari total komponen nilai tugas mata kuliah                        |
| Batas waktu        | Senin, 12 Oktober 2026, pukul 23.59 WIB                                  |
| Platform komputasi | Google Colab; runtime GPU T4 gratis sudah memadai                        |
| Kanal pengumpulan  | Thread forum LMS kelas                                                   |
| Rujukan materi     | Bahan tayang Sesi 05 — _Advanced CNNs and Transfer Learning_, Slide 5–30 |

### Luaran yang Dikumpulkan

1. **Jupyter Notebook (`.ipynb`)** yang telah dieksekusi lengkap beserta seluruh output sel.
2. **Dokumen laporan analisis (`.pdf` atau `.md`)** sepanjang 3–5 halaman.
3. **Berkas catatan metrik mentah (`.csv` atau `.json`)**.

---

# 1. Latar Belakang dan Konteks Masalah

Pada Sesi 05, kita telah mempelajari komponen penting yang memungkinkan arsitektur _Convolutional Neural Networks_ (CNN) modern dilatih secara stabil dan menghasilkan performa tinggi:

1. **Batch Normalization (Slide 6–8):** Menstabilkan distribusi aktivasi dan membantu mempercepat konvergensi.
2. **Residual Connections / Skip Connections (Slide 10–12):** Mengatasi masalah degradasi performa pada jaringan yang sangat dalam.
3. **Transfer Learning dan Fine-Tuning (Slide 14–18):** Memanfaatkan representasi visual dari model _pretrained_.
4. **Data Augmentation (Slide 20–22):** Bertindak sebagai mekanisme regularisasi untuk membantu mengurangi _overfitting_.

Sesi ini juga membahas arsitektur modern seperti EfficientNet (Slide 24) serta panduan praktis dalam memilih strategi adaptasi domain (Slide 27–29).

Tugas ini menyimulasikan kasus rekayasa praktis yang lazim dijumpai dalam industri maupun riset terapan _computer vision_: membangun sistem klasifikasi citra dengan akurasi tinggi meskipun jumlah data berlabel terbatas (_small-data regime_) dan anggaran komputasi minimal.

Mahasiswa dituntut merumuskan rekomendasi teknis berdasarkan bukti empiris dari eksperimen mandiri untuk menentukan strategi yang paling sesuai diterapkan pada lingkungan produksi.

Fokus utama penyelidikan berakar pada dilema teknis di Slide 15–16:

> Kapan kita cukup membekukan representasi backbone (_feature extraction_), dan dalam kondisi apa penyesuaian bobot secara mendalam (_fine-tuning_) diperlukan?

---

# 2. Tujuan dan Sasaran Tugas

Melalui tugas ini, mahasiswa diharapkan mampu:

1. Mengembangkan pipeline klasifikasi citra terstandardisasi berbasis PyTorch dan torchvision.
2. Mengevaluasi performa model secara komparatif melalui tiga paradigma:
   - Pelatihan dari nol (_from scratch_).
   - Ekstraksi fitur (_feature extraction_).
   - _Progressive fine-tuning_.
3. Menganalisis peran uji kontrol _data augmentation_ dalam menekan _generalization gap_ pada data terbatas.
4. Mengeksplorasi _trade-off_ antara efisiensi komputasi dan performa pada arsitektur modern, seperti ResNet-18, EfficientNet-B0, dan MobileNetV3.
5. Menyusun laporan analisis komparatif dan analisis kesalahan (_failure analysis_) yang kritis serta dapat dipertanggungjawabkan secara ilmiah.

---

# 3. Bagian A — Analisis Konseptual (Bobot: 20%)

Jawab kelima pertanyaan berikut secara ringkas, padat, dan terstruktur, dengan panjang **100–150 kata per nomor**.

Cantumkan nomor slide yang relevan serta sitasi literatur ilmiah pendukung apabila merujuk sumber eksternal.

## 3.1. Efek Regularisasi pada Batch Normalization (Slide 6–8)

Slide 8 menyebutkan bahwa Batch Normalization _"acts like mild regularization"_ dan _"often eliminates the need for Dropout"_.

Jelaskan mekanisme statistik mini-batch yang menimbulkan efek regularisasi tersebut.

Uraikan kondisi atau skenario spesifik ketika asumsi bahwa Batch Normalization dapat menggantikan Dropout tidak lagi berlaku atau justru dapat memicu instabilitas pelatihan.

## 3.2. Degradasi Jaringan Dalam dan Formulasi Residual (Slide 10–12)

Jawab dua pertanyaan berikut:

- Mengapa penambahan kedalaman lapisan pada _plain CNN_ (tanpa sambungan residual) justru dapat menurunkan akurasi pelatihan (_degradation problem_), padahal secara teoretis representasi model yang lebih dalam minimal dapat menyamai model dangkal melalui fungsi identitas (_identity mapping_)?
- Jelaskan bagaimana _skip connection_ mengubah formulasi aliran gradien saat _backpropagation_. Hubungkan mekanisme tersebut dengan mitigasi fenomena _vanishing gradient_ yang telah dibahas pada Sesi 02.

## 3.3. Feature Extraction vs. Fine-Tuning (Slide 14–16)

Bandingkan pendekatan _Feature Extraction_ (membekukan _backbone_) dengan _Fine-Tuning_ (membuka sebagian atau seluruh lapisan) berdasarkan empat dimensi berikut:

1. Jumlah parameter yang diperbarui (_trainable parameters_).
2. Kebutuhan volume data latih.
3. Beban komputasi dan memori per _epoch_.
4. Risiko rusaknya representasi fitur bawaan akibat pembaruan gradien yang agresif (_destructive gradient updates_).

Sintesiskan kriteria praktis untuk menentukan kapan masing-masing pendekatan paling tepat digunakan.

## 3.4. Kaidah Penerapan Data Augmentation (Slide 20, 21, dan 23)

Jawab dua pertanyaan berikut:

- Jelaskan pertimbangan metodologis mengapa pipeline _data augmentation_ hanya diaktifkan pada _training set_ dan harus dinonaktifkan pada _validation set_ serta _test set_.
- Berikan satu contoh teknik transformasi citra yang **tidak layak digunakan** untuk klasifikasi ras hewan peliharaan (_pet breeds_). Jelaskan alasan ketidaklayakannya berdasarkan invariansi semantik label.

## 3.5. Mitigasi Catastrophic Forgetting (Slide 28–29)

Jawab dua pertanyaan berikut:

- Uraikan bagaimana fenomena _catastrophic forgetting_, yaitu hilangnya sebagian representasi bawaan model, dapat terjadi selama proses adaptasi.
- Jelaskan dua strategi mitigasi konkret, misalnya _differential learning rate_, _gradual unfreezing_, atau _linear warmup_, beserta alasan teoretis mengapa strategi tersebut efektif.

---

# 4. Bagian B — Eksperimen Praktik (Bobot: 40%)

## 4.1. Pemilihan Dataset

Mahasiswa dapat memilih salah satu opsi dataset berikut.

### Opsi Utama: Oxford-IIIT Pet

**Sangat disarankan.**

Dataset tersedia melalui `torchvision.datasets.OxfordIIITPet` tanpa autentikasi pihak ketiga. Dataset ini mencakup 37 kelas ras kucing dan anjing dengan sekitar 200 citra per kelas.

**Aturan khusus:**

1. Ambil subset **10 kelas**, disarankan terdiri atas 5 ras kucing dan 5 ras anjing, untuk merepresentasikan skenario data terbatas dengan sekitar 2.000 citra.
2. Lakukan pembagian data internal secara _stratified_:
   - 70% untuk _training set_.
   - 15% untuk _validation set_.
   - 15% untuk _test set_.

### Opsi Alternatif: CIFAR-10

CIFAR-10 dapat digunakan untuk mengkaji ketidaksesuaian resolusi bawaan citra sebesar 32 × 32 piksel dengan model _pretrained_ ImageNet yang umumnya menggunakan resolusi input 224 × 224 piksel.

Hubungkan implikasi teknisnya dengan materi Slide 28.

### Opsi Mandiri: Dataset Domain Tertentu

Dataset dari Kaggle atau sumber eksternal diperbolehkan setelah memperoleh konfirmasi dosen terlebih dahulu.

Persyaratan:

- Maksimal 5.000 citra.
- Minimal 2 kelas klasifikasi.

## 4.2. Standarisasi Protokol Eksperimen

Agar perbandingan performa antarkondisi valid dan terkontrol secara ilmiah, terapkan protokol eksperimen berikut.

| Komponen         | Ketentuan                                                       |
| ---------------- | --------------------------------------------------------------- |
| Arsitektur utama | ResNet-18 (`torchvision.models.resnet18`)                       |
| Bobot pretrained | `ResNet18_Weights.IMAGENET1K_V1`                                |
| Dimensi input    | 224 × 224 piksel                                                |
| Normalisasi mean | `[0.485, 0.456, 0.406]`                                         |
| Normalisasi std  | `[0.229, 0.224, 0.225]`                                         |
| Batch size       | 32                                                              |
| Optimizer        | AdamW                                                           |
| Fungsi objektif  | `CrossEntropyLoss`                                              |
| Early stopping   | Memantau _validation loss_, dengan `patience` 3–5 epoch         |
| Random seed      | Tetap, misalnya `42`, untuk PyTorch, NumPy, dan Python `random` |

### Metrik yang Wajib Dicatat

Setiap eksperimen wajib mencatat:

1. Jumlah parameter yang dapat dilatih (_trainable parameters_).
2. _Learning rate_ untuk setiap kelompok parameter.
3. Jumlah epoch hingga konvergen atau berhenti berdasarkan _early stopping_.
4. Durasi pelatihan total.
5. Akurasi akhir pada _validation set_.
6. Macro F1-Score pada _test set_.

Catat pula _training loss_ dan _validation loss_ setiap epoch agar kurva pembelajaran dapat dibuat dan dianalisis.

## 4.3. Matriks Kondisi Eksperimen

Lakukan eksperimen pada kondisi berikut.

| Kode | Kondisi Eksperimen            | Deskripsi Teknis                                                                                                                                                                                                  |
| ---- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| B1   | Baseline from Scratch         | Latih ResNet-18 dari inisialisasi acak menggunakan `weights=None` pada 10 kelas. Kondisi ini menjadi tolok ukur efektivitas representasi _pretrained_.                                                            |
| B2   | Feature Extraction            | Gunakan ResNet-18 _pretrained_. Bekukan seluruh lapisan _backbone_ dengan `requires_grad=False`, ganti `model.fc` dengan `nn.Linear(512, 10)`, lalu latih hanya _classifier head_.                                |
| B3   | Progressive Fine-Tuning       | Mulai dari model _pretrained_, buka pembekuan (_unfreeze_) blok residual terakhir (`layer4`) dan lapisan `fc`. Terapkan _differential learning rate_, misalnya `lr=1e-4` untuk `layer4` dan `lr=1e-3` untuk `fc`. |
| B4   | Uji Kontrol Data Augmentation | Pilih salah satu kondisi, B2 atau B3, lalu jalankan dua skenario terpisah: tanpa augmentasi dan dengan augmentasi. Bandingkan performa serta tingkat _overfitting_.                                               |
| B5   | Arsitektur Pembanding (Bonus) | Jalankan konfigurasi terbaik menggunakan arsitektur berorientasi efisiensi, seperti EfficientNet-B0 atau MobileNetV3-Small. Bandingkan Macro F1, jumlah parameter, dan durasi komputasi dengan ResNet-18.         |

### Ketentuan B4: Uji Kontrol Data Augmentation

**Skenario 1 — Tanpa augmentasi**

Gunakan transformasi deterministik yang sesuai, seperti _resize/crop_ dan normalisasi.

**Skenario 2 — Dengan augmentasi**

Gunakan pipeline yang terdiri atas:

- Resize dan crop.
- `RandomHorizontalFlip`.
- `RandomRotation(15)`.
- `ColorJitter`.

Evaluasi perbedaan akurasi dan tingkat _overfitting_ antara kedua skenario.

## 4.4. Kerangka Kode Awal (Starter Code)

Gunakan potongan kode berikut sebagai acuan untuk inisialisasi model, konfigurasi _feature extraction_, _fine-tuning_, dan transformasi data. Lengkapi dengan pipeline dataset, _training loop_, evaluasi, serta pencatatan metrik.

### 4.4.1. Inisialisasi Model dan Feature Extraction

```python
import torch
import torch.nn as nn
from torchvision import models, transforms

weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
```

### 4.4.2. Progressive Fine-Tuning

```python
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(
    [
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ],
    weight_decay=1e-2,
)
```

### 4.4.3. Pipeline Transformasi Data

```python
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    ),
])

eval_transforms = weights.transforms()
```

### 4.4.4. Verifikasi Trainable Parameters

```python
trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Trainable Parameters: "
    f"{trainable_params:,} / {total_params:,}"
)
```

### Tips Implementasi

- Verifikasi jumlah parameter yang aktif sebelum optimizer dibentuk.
- Pada B2, `trainable_params` ResNet-18 seharusnya sekitar **5.130 parameter**, karena hanya `fc` yang dilatih untuk klasifikasi 10 kelas.
- Pada B3, pastikan parameter `layer4` dan `fc` memiliki `requires_grad=True`.
- Pastikan _data augmentation_ hanya diterapkan pada _training set_.
- Gunakan konfigurasi evaluasi yang konsisten untuk _validation set_ dan _test set_.
- Hindari _data leakage_, termasuk penggunaan _test set_ untuk memilih model atau mengatur _hyperparameter_.
- Simpan _checkpoint_ model dengan _validation loss_ terbaik untuk setiap kondisi eksperimen.

---

# 5. Bagian C — Laporan dan Analisis (Bobot: 35%)

Susun laporan teknis sepanjang **3–5 halaman** dalam Bahasa Indonesia yang formal dan lugas. Istilah teknis _machine learning_ dalam Bahasa Inggris dapat dipertahankan untuk menghindari distorsi makna.

## 5.1. C1 — Tabel Konsolidasi dan Visualisasi Pembelajaran

Sajikan tabel komparatif yang merangkum kondisi eksperimen B1 hingga B5, jika B5 dilaksanakan.

Tabel harus memuat:

- Strategi eksperimen.
- Jumlah _trainable parameters_.
- _Learning rate_.
- Epoch hingga konvergen atau berhenti.
- Durasi pelatihan.
- Akurasi validasi.
- Macro F1-Score pada _test set_.

Lengkapi laporan dengan grafik komparatif kurva _training loss_ dan _validation loss_ untuk kondisi B1, B2, dan B3.

## 5.2. C2 — Pembahasan Analisis Komparatif

Analisis fenomena empiris yang muncul dari hasil eksperimen dengan mengaitkannya pada konsep Slide 10–12, 16, 20, dan 28.

Bahas tiga pertanyaan berikut:

1. **Perbedaan performa:** Mengapa dapat terjadi disparitas performa antara pelatihan dari nol (B1) dan paradigma _transfer learning_ (B2 dan B3)?
2. **Stabilitas dan generalisasi:** Bagaimana _trade-off_ antara _feature extraction_ murni dan _fine-tuning_ bertahap pada volume data terbatas?
3. **Peran augmentasi:** Sejauh mana augmentasi membantu meredam _overfitting_ pada B4? Analisis selisih akurasi data latih dan validasi (_generalization gap_).

Pastikan interpretasi didasarkan pada hasil eksperimen yang benar-benar diperoleh, bukan asumsi mengenai hasil yang seharusnya terjadi.

## 5.3. C3 — Failure Analysis (Analisis Kasus Misklasifikasi)

Tampilkan **5–8 sampel citra dari test set** yang salah diprediksi oleh model terbaik.

Untuk setiap sampel, sertakan:

1. Citra input.
2. Label aktual (_ground truth_).
3. Prediksi model.
4. Probabilitas keyakinan (_confidence score_).
5. Hipotesis penyebab kesalahan.

Contoh hipotesis penyebab kesalahan:

- Ambiguitas visual antarras.
- Sudut pengambilan citra yang ekstrem.
- Oklusi atau objek yang tertutup.
- Pencahayaan yang buruk.
- Bias terhadap latar belakang.

Hipotesis harus didukung oleh pengamatan terhadap citra dan hasil prediksi. Jangan menyatakan penyebab sebagai fakta apabila bukti yang tersedia belum mencukupi.

## 5.4. C4 — Rekomendasi Rekayasa dan Batasan Eksperimen

Berikan rekomendasi praktis bagi tim rekayasa industri mengenai strategi pelatihan yang paling sesuai untuk skenario data terbatas.

Rekomendasi harus mencakup:

- Strategi yang dipilih berdasarkan hasil eksperimen.
- Alasan pemilihan berdasarkan metrik performa dan efisiensi komputasi.
- Kondisi berlakunya rekomendasi (_operational constraints_).
- Keterbatasan eksperimen.

Contoh keterbatasan yang dapat dibahas:

- Pengujian hanya menggunakan satu _random seed_.
- Jumlah kelas yang terbatas.
- Anggaran komputasi.
- Variasi dataset yang belum cukup luas.
- Ketidakpastian generalisasi ke domain atau distribusi data lain.

## 5.5. C5 — Eksplorasi Tambahan (Opsional)

Lakukan visualisasi representasi fitur, misalnya _feature map activation_ atau Grad-CAM, pada blok `layer4` sebelum dan sesudah _fine-tuning_.

Analisis bagaimana representasi visual berubah setelah model disesuaikan dengan domain data spesifik. Hubungkan temuan dengan _best practice_ pada Slide 29.

---

# 6. Luaran dan Ketentuan Pengumpulan

## 6.1. Berkas yang Wajib Dikumpulkan

| No. | Berkas                                 | Persyaratan                                                                                                                                  |
| --- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Jupyter Notebook (`.ipynb`)            | Kode bersih, modular, dapat direproduksi dari awal hingga akhir, berjalan tanpa error, serta menyertakan seluruh visualisasi dan output sel. |
| 2   | Dokumen laporan (`.pdf` atau `.md`)    | Laporan sepanjang 3–5 halaman yang memuat visualisasi, tabel metrik, dan analisis mendalam.                                                  |
| 3   | Log hasil mentah (`.csv` atau `.json`) | Catatan metrik eksperimen sebagai bukti integritas hasil.                                                                                    |
| 4   | Repositori GitHub                      | Repositori yang rapi, dilengkapi `README.md` dan `requirements.txt`.                                                                         |

## 6.2. Format dan Tata Cara Pengumpulan

1. Tugas bersifat **individual**.
2. Arsipkan seluruh berkas luaran ke dalam satu file ZIP.
3. Gunakan format penamaan:

   `Tugas05_<NIM>_<NamaMahasiswa>.zip`

4. Kumpulkan melalui thread forum LMS mata kuliah.
5. Gunakan judul balasan yang sama:

   `Tugas05_<NIM>_<NamaMahasiswa>`

---

# 7. Integritas Akademik dan Kebijakan Penggunaan AI (Bobot: 5%)

Integritas dan kejujuran ilmiah merupakan nilai mendasar dalam perkuliahan ini.

## 7.1. Pemahaman Mandiri

Setiap mahasiswa wajib memahami secara mendalam setiap baris kode dan narasi yang dikumpulkan.

Dosen berhak melakukan wawancara lisan singkat secara acak untuk menguji pemahaman mahasiswa terhadap materi dan implementasi tugas.

## 7.2. Sitasi dan Atribusi

Kode dari pustaka pihak ketiga, tutorial publik, atau referensi paper wajib mencantumkan sumber secara eksplisit.

## 7.3. Kebijakan Pemanfaatan Asisten AI

Penggunaan model AI generatif, seperti ChatGPT, Claude, GitHub Copilot, dan Gemini, **diperbolehkan** sebagai alat bantu untuk:

- Debugging kode.
- Referensi sintaksis.
- Klarifikasi pemahaman konsep.

Penggunaan AI tidak mengurangi nilai selama dilakukan secara transparan dan bertanggung jawab.

Jika menggunakan AI, sertakan lampiran **Pernyataan Penggunaan AI (_AI Disclosure Statement_)** pada akhir laporan yang menjelaskan:

1. Nama model atau alat AI yang digunakan.
2. Ruang lingkup bantuan AI dalam pengerjaan tugas.
3. Cara memverifikasi kebenaran keluaran AI secara mandiri.

Seluruh analisis, interpretasi hasil empiris, dan penarikan kesimpulan dalam laporan wajib ditulis secara mandiri.

Praktik plagiarisme, fabrikasi data, atau penyerahan kode maupun teks generik tanpa pemahaman dapat mengakibatkan pembatalan nilai.

---

# 8. Rubrik Penilaian

| No. | Kriteria Evaluasi                            |    Bobot | Indikator Kinerja Utama                                                                                                                                                                                             | Capaian Pembelajaran |
| --- | -------------------------------------------- | -------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| 1   | Analisis Konseptual (Bagian A)               |      20% | Ketepatan dan kedalaman penjelasan mekanisme BatchNorm, degradasi residual, perbandingan feature extraction dan fine-tuning, prinsip augmentasi, serta mitigasi catastrophic forgetting. Rujukan nomor slide tepat. | LO3, LO4             |
| 2   | Kualitas Implementasi Pipeline (B1–B4)       |      25% | Notebook berjalan end-to-end tanpa error. Pembekuan dan pembukaan parameter dilakukan dengan benar. Augmentasi diterapkan secara tepat pada data latih. Tidak terjadi data leakage. Random seed tercatat.           | LO1, LO2             |
| 3   | Kelengkapan Matriks Eksperimen               |      15% | Empat kondisi utama (B1–B4) terlaksana secara lengkap dan setara di bawah protokol eksperimen terstandardisasi. Kondisi B5 dieksekusi dengan baik sebagai nilai tambah.                                             | LO1, LO2             |
| 4   | Kedalaman Analisis dan Pembahasan (Bagian C) |      25% | Tabel konsolidasi metrik lengkap, kurva pelatihan terbaca jelas, failure analysis memuat hipotesis teknis yang logis, serta rekomendasi rekayasa didukung data empiris dan mencakup batasan operasional.            | LO3, LO4             |
| 5   | Struktur Laporan dan Reproduksibilitas       |      10% | Sistematika dokumen rapi, visualisasi profesional, tata bahasa lugas dan baku, konfigurasi eksperimen terdokumentasi lengkap, serta sitasi valid.                                                                   | LO1, LO4             |
| 6   | Integritas Akademik dan Pernyataan AI        |       5% | Transparansi pernyataan penggunaan AI, konsistensi angka metrik dalam laporan dengan log mentah, serta penguasaan materi tugas.                                                                                     | LO4                  |
|     | **Total**                                    | **100%** |                                                                                                                                                                                                                     |                      |

## Catatan Penilaian Data Empiris

Seluruh klaim performa dalam laporan harus bersumber dari berkas log mentah (`.csv` atau `.json`) yang dilampirkan.

Setiap kondisi eksperimen yang dilaporkan tanpa disertai data log pendukung hanya akan dinilai maksimal **50% dari bobot kriteria terkait**, karena angka tanpa artefak verifikasi tidak dapat divalidasi.

---

**Akhir dokumen**
