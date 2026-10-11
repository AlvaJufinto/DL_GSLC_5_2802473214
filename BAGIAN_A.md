<!-- @format -->

# Advanced CNNs and Transfer Learning — Concept Summary

## 1. Regularization Effect of Batch Normalization

**Referensi: Slide 6–8**

### Statistical Mechanism

Batch Normalization (BatchNorm) menormalisasi activation values menggunakan mean dan variance yang dihitung dari setiap mini-batch. Hasil normalisasi tersebut kemudian disesuaikan menggunakan learnable parameters, yaitu \(\gamma\) dan \(\beta\).

Proses normalisasi dirumuskan sebagai berikut:

\[
\hat{x}\_i = \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
\]

\[
y_i = \gamma\hat{x}\_i+\beta
\]

Keterangan:

- \(x_i\): input activation.
- \(\mu_B\): mean dari mini-batch.
- \(\sigma_B^2\): variance dari mini-batch.
- \(\epsilon\): small constant untuk numerical stability.
- \(\gamma\) dan \(\beta\): learnable scale and shift parameters.

Karena mean dan variance dapat berbeda antar-mini-batch, BatchNorm memperkenalkan variasi kecil selama training. Variasi ini dapat memberikan efek _mild regularization_ yang membantu model melakukan generalisasi terhadap unseen data.

Dalam beberapa kasus, regularization effect dari BatchNorm dapat mengurangi kebutuhan terhadap Dropout. Namun, BatchNorm tidak selalu dapat menggantikan Dropout karena efektivitasnya bergantung pada model architecture, dataset characteristics, dan hasil evaluasi.

### Instability with Small Batch Sizes

BatchNorm bergantung pada estimasi statistik dari setiap mini-batch. Ketika batch size terlalu kecil, estimasi mean dan variance dapat menjadi kurang stabil atau kurang representatif.

Akibatnya, training stability dan model performance dapat terpengaruh. Oleh karena itu, regularization effect dari BatchNorm tidak berarti Dropout selalu dapat dihilangkan.

**Additional note:** Pembahasan mengenai instability pada small batch sizes merupakan penjelasan tambahan. BatchNorm juga membantu mengurangi sensitivitas training terhadap parameter initialization.

## 2. Degradation Problem and Residual Connections

**Referensi: Slide 10–12**

### Degradation Problem in Deep Networks

Menambah jumlah layers tidak selalu meningkatkan model performance. Pada plain neural networks yang semakin dalam, optimization menjadi lebih sulit dan training accuracy bahkan dapat mengalami penurunan. Fenomena ini dikenal sebagai _degradation problem_.

Deep networks juga dapat menghadapi:

- **Vanishing gradient:** Gradien menjadi sangat kecil sehingga pembaruan parameter pada early layers menjadi kurang efektif.
- **Exploding gradient:** Gradien menjadi sangat besar sehingga pembaruan parameter dapat menjadi tidak stabil.

Kedua masalah tersebut dapat menghambat proses optimization, meskipun degradation problem dan gradient problems merupakan fenomena yang berbeda.

### Residual Formulation

Residual connection atau _skip connection_ diperkenalkan untuk mempermudah training pada deep neural networks. Mekanisme ini menambahkan input awal secara langsung ke output dari sebuah transformation block.

Persamaan residual sederhana:

\[
y = F(x)+x
\]

Keterangan:

- \(x\): input ke residual block.
- \(F(x)\): transformation yang dipelajari oleh layers di dalam block.
- \(y\): output setelah transformation ditambahkan ke input awal.

### Improving Gradient Flow

Identity shortcut \(+x\) menyediakan jalur tambahan bagi gradients selama proses backpropagation. Dengan jalur ini, gradients dapat mengalir melalui network tanpa harus sepenuhnya melewati seluruh transformations di dalam residual block.

Mekanisme tersebut membantu optimization pada deep networks dan dapat mengurangi kesulitan gradient propagation. Namun, residual connections tidak menjamin seluruh vanishing gradient atau optimization problems akan hilang.

## 3. Feature Extraction vs. Fine-Tuning

**Referensi: Slide 14–16 dan 29**

### Feature Extraction

Pada feature extraction, pretrained layers dibekukan menggunakan `requires_grad=False` atau mekanisme freezing yang sesuai dengan framework yang digunakan.

Hanya bagian classifier head yang baru dilatih, sedangkan pretrained weights tetap tidak berubah.

Karakteristik:

- **Trainable parameters:** Relatif sedikit.
- **Computational cost:** Lebih rendah.
- **Memory usage:** Umumnya lebih rendah karena gradients dan optimizer states tidak perlu disimpan untuk frozen parameters.
- **Main purpose:** Memanfaatkan pretrained feature representations untuk menyelesaikan task baru.

Feature extraction cocok sebagai baseline, terutama ketika jumlah training data terbatas dan pretrained model sudah memiliki feature representations yang relevan.

### Fine-Tuning

Pada fine-tuning, sebagian atau seluruh pretrained layers di-unfreeze agar weights dapat diperbarui berdasarkan dataset target.

Karakteristik:

- **Trainable parameters:** Lebih banyak.
- **Computational cost:** Umumnya lebih tinggi.
- **Memory usage:** Meningkat karena gradients dan optimizer states diperlukan untuk trainable parameters.
- **Main purpose:** Menyesuaikan pretrained representations dengan karakteristik dataset baru.

Fine-tuning memberikan model lebih banyak fleksibilitas untuk beradaptasi, tetapi proses update yang terlalu agresif dapat merusak useful pretrained representations.

### Data Availability and Computational Cost

Transfer learning sangat bermanfaat ketika jumlah training data terbatas karena model memanfaatkan knowledge yang telah diperoleh dari dataset sebelumnya.

Melatih model besar dari scratch umumnya membutuhkan lebih banyak data, training time, dan computational resources.

Feature extraction biasanya lebih ringan dibandingkan fine-tuning karena hanya classifier head yang diperbarui. Sebaliknya, fine-tuning membutuhkan tambahan computation dan memory untuk memperbarui pretrained layers yang di-unfreeze.

### Practical Strategy

Strategi yang umum digunakan adalah:

1. Mulai dengan **feature extraction** untuk membangun baseline.
2. Evaluasi validation performance.
3. Jika performa masih dapat ditingkatkan, lanjutkan dengan **fine-tuning**.
4. Gunakan lower learning rate untuk pretrained layers.
5. Terapkan learning-rate scheduling dan monitor validation metrics.

Pendekatan ini memungkinkan model beradaptasi secara bertahap tanpa langsung mengubah seluruh pretrained weights.

## 4. Data Augmentation Guidelines

**Referensi: Slide 20–21**

### Training-Only Augmentation

Data augmentation merupakan teknik untuk menghasilkan variasi training examples melalui transformasi gambar, seperti flipping, rotation, cropping, dan color adjustment.

Tujuannya adalah memperluas variasi visual yang dilihat model selama training sehingga model tidak terlalu bergantung pada karakteristik spesifik dari training samples.

Augmentation sangat berguna ketika dataset berukuran kecil karena dapat membantu meningkatkan generalization performance.

Penerapannya harus dibedakan berdasarkan dataset split:

- **Training set:** Menggunakan random augmentation yang sesuai dengan task.
- **Validation set:** Menggunakan preprocessing yang konsisten, tanpa random augmentation yang mengubah gambar.
- **Test set:** Menggunakan preprocessing yang konsisten dengan evaluation pipeline, tanpa random augmentation kecuali memang dirancang sebagai bagian dari protokol evaluasi.

Dengan demikian, validation dan test performance dapat diukur pada data yang tidak mengalami random distortion.

### Avoid Excessive Transformations

Untuk pet breed classification, augmentation harus mempertahankan semantic information yang relevan untuk membedakan setiap breed.

Transformasi yang terlalu ekstrem dapat merusak informasi penting:

- **Extreme color jitter:** Dapat mengubah warna atau coat patterns yang merupakan distinctive features dari breed tertentu.
- **Excessive cropping:** Dapat menghilangkan bagian tubuh atau ciri khas yang dibutuhkan untuk classification.
- **Excessive zoom:** Dapat mengubah framing gambar sehingga informasi penting tidak lagi terlihat.

Gunakan realistic augmentation yang merepresentasikan variasi yang mungkin ditemukan pada real-world data.

Prinsip utamanya adalah meningkatkan variasi training data tanpa mengubah semantic label atau menghilangkan fitur visual yang relevan.

## 5. Mitigating Catastrophic Forgetting

**Referensi: Slide 16 dan 28–29**

### What Is Catastrophic Forgetting?

_Catastrophic forgetting_ dapat terjadi ketika fine-tuning mengubah pretrained weights secara terlalu agresif. Akibatnya, useful feature representations yang telah dipelajari dari large-scale datasets, seperti ImageNet, dapat terganggu atau hilang sebagian.

Fenomena ini dapat terjadi ketika learning rate terlalu tinggi, terlalu banyak layers langsung di-unfreeze, atau training dilakukan tanpa monitoring yang memadai.

### Mitigation Strategies

#### 1. Gradual Unfreezing

Gradual unfreezing adalah strategi membuka pretrained layers secara bertahap, bukan langsung melatih seluruh network sekaligus.

Sebagai contoh:

1. Latih classifier head dengan pretrained layers tetap frozen.
2. Unfreeze sebagian layers yang lebih dalam.
3. Lakukan fine-tuning menggunakan lower learning rate.
4. Evaluasi validation performance sebelum membuka lebih banyak layers.

Strategi ini memberikan kontrol yang lebih baik terhadap perubahan pretrained representations selama fine-tuning.

#### 2. Lower and Differential Learning Rates

Gunakan lower learning rate pada pretrained layers untuk membatasi besarnya weight updates.

Jika diperlukan, gunakan _differential learning rates_, yaitu learning rates yang berbeda untuk bagian model yang berbeda.

Sebagai contoh:

- **Classifier head:** Learning rate relatif lebih tinggi karena bagian ini perlu mempelajari pemetaan baru ke target classes.
- **Pretrained layers:** Learning rate lebih rendah untuk mempertahankan useful representations yang sudah dipelajari.

Pendekatan ini memungkinkan classifier head beradaptasi lebih cepat, sementara pretrained representations diperbarui secara lebih hati-hati.

#### 3. Learning-Rate Scheduler

Learning-rate scheduler menyesuaikan learning rate selama training berdasarkan jadwal atau kondisi tertentu.

Scheduler dapat membantu mengendalikan ukuran parameter updates selama fine-tuning. Misalnya, learning rate dapat diturunkan secara bertahap agar proses optimization menjadi lebih halus ketika training berlangsung.

Namun, penggunaan scheduler tidak menjamin catastrophic forgetting tidak akan terjadi. Hasilnya tetap bergantung pada learning-rate configuration, unfreezing strategy, dataset characteristics, dan validation performance.

### Key Takeaway

Untuk mengurangi risiko catastrophic forgetting, lakukan fine-tuning secara bertahap, gunakan lower learning rate pada pretrained layers, dan monitor validation metrics. Kombinasi strategi tersebut membantu menjaga useful pretrained representations sekaligus memberikan kesempatan bagi model untuk beradaptasi terhadap dataset target.
