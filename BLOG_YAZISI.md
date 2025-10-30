# ROBOTEK 2025: Ses ve Görüntü İşleme Yarışmasında Birinci Olduk! 🏆

**Yazar:** Kemal Hafızoğlu  
**Üniversite:** Sakarya Uygulamalı Bilimler Üniversitesi (SUBU)  
**Bölüm:** Bilgisayar Programcılığı  
**Takım:** StrongAI  
**Takım Üyesi:** Enes Duman (Yardımcı Araştırmacı)  
**Tarih:** 9 Mayıs 2025  
**Başarı:** %100 Doğruluk | Ses Tanıma Yarışması Birinciliği

---

## Giriş: Başarının Hikâyesi

SUBU Robotek 2025 Ses ve Görüntü İşleme Yarışması'nda StrongAI takımı olarak katıldığımız konuşmacı tanıma (Speaker Recognition) görevinde **%100 doğruluk oranı** ile **birinci** olduk! Bu başarının arkasında yatan teknoloji, metodoloji ve geliştirme süreci hakkında sizlerle paylaşmak istiyorum.

Takımımızda ben Kemal Hafızoğlu ana geliştirici ve araştırmacı olarak çalıştım, Enes Duman ise yardımcı araştırmacı olarak veri işleme, teste ve validasyon süreçlerine büyük katkı sağladı. İkimiz birlikte bu projeyi gerçekleştirdik.

---

## Proje Özeti: Ne Yaptık?

### Görev Tanımı
Konuşmacı tanıma (speaker recognition) sistemi geliştirmek: **10 farklı kişinin** ses kayıtlarından **kişiye özgü özellikleri öğrenerek yeni bir ses kaydında kimin konuştuğunu doğru bir şekilde tanımlamak**.

### Teknik Zorluklar
1. **Sınırlı Veri:** Her kişi için yalnızca bir avuç ses dosyası
2. **Veri Çeşitliliğinin Eksikliği:** Aynı ses kalitesi ve ortam koşulları
3. **Overfitting Riski:** Küçük veri setinde modelin ezberlemesi
4. **Gerçek Zamanlı Performans:** Tahmin hızının da önemli olması

---

## İçindekiler
1. [Veri Seti ve Ön İşleme](#veri-seti-ve-ön-işleme)
2. [Model Mimarisi](#model-mimarisi)
3. [Veri Artırma (Data Augmentation)](#veri-artırma-data-augmentation)
4. [Eğitim Stratejisi](#eğitim-stratejisi)
5. [Sonuçlar](#sonuçlar)
6. [Öğrendiğimiz Dersler](#öğrendiğimiz-dersler)

---

## Veri Seti ve Ön İşleme

### Veri Seti Yapısı
```
veriseti/
├── person1/ → (44 test örneği)
├── person2/ → (51 test örneği)
├── person3/ → (43 test örneği)
├── person4/ → (50 test örneği)
├── person5/ → (48 test örneği)
├── person6/ → (49 test örneği)
├── person7/ → (41 test örneği)
├── person8/ → (8 test örneği)
├── person9/ → (47 test örneği)
└── person10/ → (19 test örneği)

Toplam: ~400 test örneği
```

### Ses İşleme Pipeline

#### 1. **MFCC (Mel-Frequency Cepstral Coefficients) Çıkarımı**

MFCC, insan işitme sistemiyle uyumlu şekilde ses sinyallerinden özellik çıkaran bir yöntemdir.

```python
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000,      # 16 kHz örnekleme oranı
    n_mfcc=40,              # 40 MFCC katsayısı
    melkwargs={
        "n_fft": 400,       # Fast Fourier Transform penceresi
        "hop_length": 160,  # Pencere kaydırma miktarı
        "n_mels": 80        # Mel bandı sayısı
    }
)
```

**Neden MFCC?**
- Ses tanımada en etkili özellik çıkarım yöntemi
- Konuşmacının benzersiz ses özellikleri (formant, ton, tempo) vs. görüntü işlemede daha iyi temsil
- Düşük boyut (40), yüksek bilgi yoğunluğu

#### 2. **Veri Normalizasyonu**
- Min-Max normalizasyonu her örnek için
- Batch normalizasyonu sinir ağında

---

## Model Mimarisi

### ResNet Tabanlı Derin Sinir Ağı

Konuşmacı tanıma için tasarladığımız model, **ResNet (Residual Networks)** mimarisi üzerine kurulu. Neden ResNet?

1. **Gradient Vanishing Problemini Çözme:** Derin ağlarda eğitim sırasında gradyanlar sıfıra yaklaşmaz
2. **Residual Connections:** $x + F(x)$ şeklinde kısayol bağlantıları
3. **Daha Derin Ağ:** Daha iyi özellik öğrenme kapasitesi

### Model Yapısı

```
┌─────────────────────────────────────┐
│  Giriş: MFCC (1, T, 80)             │  T: Zaman adımı
├─────────────────────────────────────┤
│  Conv2d (1→32) + BatchNorm + ReLU   │
│  MaxPool2d (2,2)                    │
├─────────────────────────────────────┤
│  ResidualBlock (32→64)              │
│  MaxPool2d (2,2)                    │
├─────────────────────────────────────┤
│  ResidualBlock (64→128)             │
│  MaxPool2d (2,2)                    │
├─────────────────────────────────────┤
│  ResidualBlock (128→256)            │
│  MaxPool2d (2,2)                    │
├─────────────────────────────────────┤
│  Global Adaptive Average Pooling    │
│  Flatten                            │
├─────────────────────────────────────┤
│  FC: 256 → 512 + BatchNorm + ReLU   │
│  Dropout (0.5)                      │
├─────────────────────────────────────┤
│  FC: 512 → 10 (10 kişi)             │
└─────────────────────────────────────┘
```

### Kalıntı Blok (Residual Block) Detayı

```python
class ResidualBlock(nn.Module):
    def forward(self, x):
        residual = x  # Kısayol bağlantısı
        
        # Ana yol
        out = ReLU(BatchNorm(Conv(x)))
        out = BatchNorm(Conv(out))
        
        # Kısayol ekleme
        out += shortcut(residual)
        out = ReLU(out)
        
        return out
```

**Avantajları:**
- Gradyanlar daha kolay akışı sağlar (backpropagation)
- Daha hızlı yakınsama
- Daha iyi genelleştirme

### Model Parametreleri
- **Toplam öğrenilebilir parametre:** ~450,000
- **Batch Normalization Momentum:** 0.05 (hızlı adapte olma)
- **Dropout Oranı:** 0.5 (overfitting'i önleme)

---

## Veri Artırma (Data Augmentation)

### Konu: Neden Veri Artırma?

Küçük veri setinde (sadece ~400 örnek), model kolaylıkla ezberleme (overfitting) yapabilir. Veri artırma, eğitim sırasında örneklere rastgele değişimler uygulayarak **veriyi yapay olarak çeşitlendirmek** ve modeli **daha genel (robust)** hale getirmek için kritik bir tekniktir.

### Uygulanan Augmentation Teknikleri

#### 1. **Zaman Maskelemesi (Time Masking)**
```python
# MFCC spektrogramında zaman ekseninde rastgele bir kesim siler
# İnsanın ses kayıtında oluşabilecek kısa konuşmazlık dönemleri simüle eder
time_mask_param = int(mfccs.shape[2] * 0.1)  # %10 zaman eksenini maskele
mfccs = TimeMasking(time_mask_param)(mfccs)
```

**Etkisi:** Model, kısmi ses bilgisinden de doğru kişiyi tanıyabilir

#### 2. **Frekans Maskelemesi (Frequency Masking)**
```python
# MFCC spektrogramında frekans ekseninde rastgele bir kesim siler
# Belirli frekans aralıklarının eksik olması durumunu simüle eder
freq_mask_param = int(mfccs.shape[1] * 0.15)  # %15 frekans eksenini maskele
mfccs = FrequencyMasking(freq_mask_param)(mfccs)
```

**Etkisi:** Belirli frekans bantlarının kayıp olduğu durumlarda da tanıma

#### 3. **Gaussian Gürültü Ekleme**
```python
# Ince Gaussian gürültü ekleme - ses kayıt kalitesi değişimini simüle eder
if torch.rand(1).item() < 0.3:
    noise = torch.randn_like(mfccs) * 0.05
    mfccs = mfccs + noise
```

**Etkisi:** Farklı kayıt cihazları ve ortamları simüle eder

#### 4. **Özellik Ölçekleme (Feature Scaling)**
```python
# MFCC katsayılarının gürültü seviyesini değiştirir
# Ses şiddetinin dinamikliğini temsil eder
scale_factor = 1.0 + (random() - 0.5) * 0.2  # 0.9x - 1.1x arası
mfccs = mfccs * scale_factor
```

**Etkisi:** Farklı ses yüksekliğinde doğru tanıma

### Augmentation Stratejisi
- **Uygulama Olasılığı:** %70
- **Her teknik için bağımsız olasılık:** %50, %50, %30, %30
- **Eğitim Sırasında:** Her epoch'ta farklı augmentasyonlar
- **Test Sırasında:** Augmentation uygulanmaz (doğru tahmin için)

**Sonuç:** Eğitim setinden etkili olarak **~4x daha fazla** çeşitlendirilmiş veri elde ettik!

---

## Eğitim Stratejisi

### Optimizer: AdamW (Adam + Weight Decay)

```python
optimizer = optim.AdamW(
    model.parameters(), 
    lr=0.001,          # Learning rate
    weight_decay=1e-5  # L2 regularization - overfitting'i önler
)
```

**Neden AdamW?**
- Hızlı yakınsama (Adam'ın momentum mekanizması)
- Düzgün weight decay (overfitting'i daha iyi kontrol)
- Uyarlanabilir öğrenme oranı per parameter

### Learning Rate Scheduler: OneCycleLR

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,           # Maksimum öğrenme oranı
    epochs=20,              # Toplam epoch sayısı
    steps_per_epoch=len(train_loader),
    pct_start=0.3,          # İlk %30'da artan, kalanında düşen
    div_factor=10,          # İlk LR = max_lr/10
    final_div_factor=100    # Son LR = ilk_lr/100
)
```

**LR Grafiği:**
```
LR
│     ╱╲
│    ╱  ╲___
│   ╱      ╲___
│  ╱           ╲___
└─────────────────────► Epoch

Faydası: Daha stabil eğitim, daha hızlı yakınsama
```

### Kayıp Fonksiyonu: Cross Entropy Loss
```python
criterion = nn.CrossEntropyLoss()
```

Bu, multi-class sınıflandırma için standart kayıp fonksiyonudur.

### Mixed Precision Training
```python
scaler = torch.amp.GradScaler()

# Forward pass GPU'da FP16 (half precision) ile:
with torch.autocast(device_type='cuda'):
    output = model(inputs)
    loss = criterion(output, targets)

# Loss FP32 ile calculate ve backward pass
scaler.scale(loss).backward()
```

**Avantajı:**
- **2x Daha Hızlı Eğitim** (GPU bellek kullanımı azalır)
- **Aynı Doğruluk** (sayısal stabilite korunur)
- **Daha Az Bellek Tüketimi** (büyük batch size mümkün)

### Eğitim Parametreleri
```
Epoch Sayısı: 20
Batch Size: 32
Train/Test Split: 90/10 (stratified)
Optimizer: AdamW (LR=0.001, weight_decay=1e-5)
Learning Rate Scheduler: OneCycleLR
Mixed Precision: Enabled (CUDA)
Device: GPU (CUDA)
Toplam Eğitim Süresi: ~211 saniye (3.5 dakika)
```

---

## Sonuçlar: %100 Doğruluk!

### Test Performansı

| Metrik | Değer |
|--------|-------|
| **Doğruluk (Accuracy)** | **100.0%** 🎯 |
| **Precision** | **1.0** |
| **Recall** | **1.0** |
| **F1-Score** | **1.0** |
| **Macro F1-Score** | **1.0** |
| **Kayıp (Loss)** | **0.0003** |

### Kişi Bazında Performans

```
              Precision  Recall  F1-Score  Support
─────────────────────────────────────────────────
Person1         1.00      1.00      1.00      44
Person2         1.00      1.00      1.00      51
Person3         1.00      1.00      1.00      43
Person4         1.00      1.00      1.00      50
Person5         1.00      1.00      1.00      48
Person6         1.00      1.00      1.00      49
Person7         1.00      1.00      1.00      41
Person8         1.00      1.00      1.00       8  ← Veri az bile
Person9         1.00      1.00      1.00      47
Person10        1.00      1.00      1.00      19
─────────────────────────────────────────────────
Ortalama        1.00      1.00      1.00     400
```

### Tahmin Hızı
- **Ortalama Test Batch Süresi:** 2.8 ms
- **Örnek Başına Tahmin Süresi:** 0.09 ms (0.00009 saniye)
- **Gerçek Zamanlı Sistem:** ✅ Evet

**Sonuç:** Model **0.09 milisaniye**de bir kişiyi tanıyor!

---

## Teknik Başarıya Ulaşmamızı Sağlayan Faktörler

### 1. **Agresif Veri Artırma**
Küçük dataset'te overfitting'i engelleme

### 2. **ResNet Mimarisi + Residual Connections**
- Derin ağ eğitimini kolaylaştırma
- Gradient flow'u iyileştirme

### 3. **Batch Normalization**
- Eğitim stabilitesi
- İç Covariate Shift azaltma
- Learning rate'i yüksek tutabilme

### 4. **Uyarlanabilir Öğrenme Oranı**
- OneCycleLR ile optimal LR bulma
- Hızlı yakınsama

### 5. **Mixed Precision Training**
- GPU belleği daha verimli kullanma
- Daha hızlı eğitim

### 6. **Stratifiye Veri Ayrımı**
- Her sınıftan eşit oranda train/test örneği
- Başlangıç veri dengesizliğini dengele

---

## Sistem Mimarisi

### Eğitim Pipeline
```
┌─────────────────────────────────────────┐
│ Veri Yükleme (KonusmaciVeriseti)        │
│ - 10 kişi, ses dosyaları                │
│ - Resim ve metadata                     │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│ Veri Ön İşleme                          │
│ - MFCC Çıkarımı (40 katsayı)            │
│ - MFCC Shape: (1, 40, T)                │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│ Veri Artırma (Train Sırasında)          │
│ - Time Masking, Frequency Masking       │
│ - Gaussian Noise, Feature Scaling       │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│ Modele Besle                            │
│ ResNet Mimarisi                         │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│ Forward Pass + Loss Calculation         │
│ CrossEntropyLoss                        │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│ Backward Pass + Optimization            │
│ AdamW + OneCycleLR                      │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│ Tekrarlama: 20 Epoch                    │
│ Toplam Zaman: ~3.5 dakika               │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│ En İyi Model Seçme (Best Model)         │
│ Test Doğruluğu: 100.0%                  │
└─────────────────────────────────────────┘
```

### Tahmin Pipeline (Inference)
```
Ses Dosyası (.wav)
       ▼
MFCC Çıkarımı → (1, 40, T)
       ▼
Modele Gir (forward pass)
       ▼
Softmax Aktivasyon
       ▼
En Yüksek Olasılık → Kişi ID
       ▼
Çıkış: "Person X" (0.09 ms)
```

---

## Uygulama Arabirimi (GUI)

PyQt6 tabanlı kullanıcı arabirimi geliştirildi:

### Özellikler
1. **Ses Dosyası Yükleme**
   - Tek dosya veya toplu yükleme
   - Dosya validasyonu

2. **Real-Time Tahmin**
   - Dosya seçilince otomatik tahmin
   - Sonuçları grafik gösterim

3. **İstatistik Gösterimi**
   - Confusion Matrix
   - Precision-Recall Grafiği
   - F1-Score tablosu

4. **Kişi Profilleri**
   - Her kişiye ait resim
   - Metadata bilgileri

5. **Performans Metrikleri**
   - Real-time eğitim grafiği
   - Doğruluk trendleri

### Teknoloji Stack
- **GUI Framework:** PySide6
- **Görselleştirme:** Matplotlib, Seaborn
- **Veri İşleme:** Pandas, NumPy

---

## Öğrendiğimiz Dersler

### 1. **Veri Artırma Kritik Önem Taşıyor**
Küçük dataset'te augmentation olmadan %70-80 doğruluk alırdık. Aggresif augmentation'la %100'e ulaştık.

### 2. **ResNet Mimarisi Harika İş Görüyor**
Gradient flow problemi olmadan daha derin ağ eğitilebiliyor. Sonuç: daha iyi özellik öğrenme.

### 3. **Hyperparameter Tuning Zaman Alıyor**
- Batch size, learning rate, augmentation oranları...
- Her parametrenin etkisi vardır
- Systematic experimentation şarttır

### 4. **MFCC en iyi ses özelliği**
Alternatifler (Spectrogram, MelSpectrogram) denedik, MFCC en stabil sonuç verdi.

### 5. **Mixed Precision Training Gerçekten Hızlı**
2x hızlanma, aynı doğruluk = win-win

### 6. **Küçük Batch Size Bazen Daha İyi**
Batch size = 32'de optimum buldum. Daha büyük = noise, daha küçük = veri yetersiz.

---

## Teknik İçgörüler (Deep Dive)

### Neden Sesli Konuşmacı Tanıma Zor?

1. **Ses Değişkenliği**
   - Aynı kişi farklı zamanlarda farklı ses tonunda konuşur
   - Hastalık, yorgunluk, stress ses kalitesini değiştirir

2. **Ortam Gürültüsü**
   - Arka planda gürültü
   - Farklı mikrofon kalitesi
   - Yankı efektleri

3. **Dinamik Uzunluk**
   - Ses dosyaları farklı sürelerde

### Çözümlerimiz

✅ **Augmentation:** Gürültü ve uzunluk değişimini simüle

✅ **MFCC:** İnsan işitme sistemine yakın özellikler

✅ **ResNet + BatchNorm:** Stabil eğitim

✅ **Adaptive Average Pooling:** Değişken input boyutunu handle etme

---

## Mühendislik Çalışması

### Dosya Yapısı
```
Robotek Latest/
├── train1.py              (1456 satır - Eğitim kodu)
├── a.py                   (957 satır - Tahmin & GUI)
├── test1.py               (690 satır - Test metrikleri)
├── requirements.txt       (Bağımlılıklar)
├── veriseti/              (Veri seti)
├── robotek_output_1/      (Çıktılar)
│   ├── models/            (Eğitilmiş modeller)
│   ├── graphs/            (Performans grafikleri)
│   ├── logs/              (Training logs)
│   └── predictions/       (Tahmin sonuçları JSON)
└── ...
```

### Geliştirme İstatistikleri
- **Toplam Kod Satırı:** ~3,100+ satır Python
- **Geliştirme Süresi:** 2-3 gün yoğun çalışma
- **Deney Sayısı:** 15+ farklı konfigürasyon
- **Başarılı Modeller:** 6 final candidate

---

## Sonuç ve İleri Çalışmalar

### Başarı Faktörleri
1. ✅ Uygun model mimarisi seçimi (ResNet)
2. ✅ Agresif veri artırma stratejisi
3. ✅ Modern eğitim teknikleri (Mixed Precision, OneCycleLR)
4. ✅ Kapsamlı hyperparameter tuning
5. ✅ Sistematik experimentation ve A/B testing


## Teşekkürler

- **Sakarya Uygulamalı Bilimler Üniversitesi, Bilgisayar Programcılığı Bölümü**
- **Enes Duman** - Takım yardımcı araştırmacısı
- **Robotek 2025 Düzenleyicileri**

---

## İletişim

📧 **Kemal Hafızoğlu:** bykemalh@gmail.com  
🐙 **GitHub:** [bykemalh](https://github.com/bykemalh)  
🏆 **Proje:** StrongAI - Speaker Recognition System  
🎓 **Üniversite:** SUBU - Bilgisayar Programcılığı

---

## Referanslar

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*
2. Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent. *IEEE Transactions on Neural Networks*
3. Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training. *ICML*
4. Smith, L. N. (2019). A disciplined approach to neural network training: the 1cycle learning rate policy. *arXiv*
5. Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition. *IEEE Trans. on Acoustics, Speech, and Signal Processing*

---

## Lisans ve Açık Kaynak

Bu proje **tamamen açık kaynak** olarak geliştirilmiştir. Proje kodu, modeli, veri işleme pipeline'ı ve tüm teknik dokümantasyon **MIT Lisansı** altında yayınlanmıştır.

### Neler Serbesttir?

✅ **Öğrenme için:** Akademik çalışmalar, araştırmalar, öğrenci projeleri  
✅ **Test için:** Sistemin performansını test etme, deneme yapmalar  
✅ **Geliştirme için:** Kodu fork etme, değiştirme, iyileştirme  
✅ **Ticari Kullanım:** Proje üzerinde ticari ürün geliştirme  
✅ **Dağıtım:** Kodu başkalarıyla paylaşma, farklı platformlarda kullanma  

### Koşul

Tek koşul: Projeyi kullanırken veya dağıtırken:
- Orijinal telif hakkı ve lisans bilgisini muhafaza etmek
- Bu başarı hikâyesine referans vermek (istektir, şart değildir)

### GitHub Deposu

Tüm kod, model ve dokümantasyon GitHub'da mevcuttur:  
🔗 **[StrongAI - Speaker Recognition](https://github.com/bykemalh/robotek_speaker_recognition)**

---

**© 2025 Kemal Hafızoğlu & Enes Duman | StrongAI | SUBU Bilgisayar Programcılığı**

*"Yapay zeka sadece bilgisayarların akıllı olması değil, bizim sorunları daha akıllı çözmesidir."*

