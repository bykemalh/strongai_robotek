import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import datetime
import csv
import logging
import copy
import base64
import io
import json
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from functools import lru_cache
from torch.cuda.amp import autocast
import shutil

# Bir sonraki müsait çıktı dizinini bulma fonksiyonu
def get_next_output_dir():
    """Bir sonraki müsait çıktı dizinini bulur"""
    base_name = "robotek_output"
    i = 1
    while os.path.exists(f"{base_name}_{i}"):
        i += 1
    return f"{base_name}_{i}"

# Çıktı dizinleri
OUTPUT_DIR = get_next_output_dir()
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")

# Takım bilgileri
YARISMA_ADI = "robotek"
TAKIM_ADI = "StrongAI"
TAKIM_ID = "team_56"

def output_dizinleri_olustur():
    for directory in [OUTPUT_DIR, MODEL_DIR, GRAPH_DIR, LOG_DIR, PREDICTION_DIR]:
        os.makedirs(directory, exist_ok=True)
    return True

# ÖNEMLİ: Dizinleri oluştur (loglama ayarlarından ÖNCE çağrılmalı)
output_dizinleri_olustur()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "robotek_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ROBOTEK")

# 1. GELİŞTİRİLMİŞ VERİ SETİ SINIFI - VERİ ARTIRMA EKLENDİ
class KonusmaciVeriseti(Dataset):
    def __init__(self, veri_klasoru, transform=True, sample_rate=16000, max_duration=5):
        self.veri_klasoru = veri_klasoru
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        
        self.ses_dosyalari = []
        self.etiketler = []
        self.resim_yollari = {}
        self.kisi_bilgileri = {}  # Kişi bilgilerini saklamak için
        self.dosya_idleri = []    # Chunk ID'leri için yeni eklenen alan
        
        # Dosya ve klasör kontrolü yapma
        if not os.path.exists(veri_klasoru):
            raise FileNotFoundError(f"Veri klasörü bulunamadı: {veri_klasoru}")
            
        try:
            # Kişi klasörlerini tarama
            kisi_klasorleri = sorted([d for d in os.listdir(veri_klasoru) if d.startswith('person')])
            
            if not kisi_klasorleri:
                raise ValueError(f"Veri klasöründe 'person' ile başlayan klasör bulunamadı: {veri_klasoru}")
                
            for i, kisi_klasoru in enumerate(kisi_klasorleri):
                kisi_yolu = os.path.join(veri_klasoru, kisi_klasoru)
                
                # Klasör mü kontrol et
                if not os.path.isdir(kisi_yolu):
                    logger.warning(f"Geçersiz kişi klasörü, atlanıyor: {kisi_yolu}")
                    continue
                    
                try:
                    kisi_id = int(kisi_klasoru.replace('person', '')) - 1  # 0-based index
                except ValueError:
                    logger.warning(f"Kişi klasörü sayısal ID içermiyor, atlanıyor: {kisi_klasoru}")
                    continue
                
                # Kişi bilgilerini sakla
                self.kisi_bilgileri[kisi_id] = {
                    'id': kisi_id,
                    'klasor_adi': kisi_klasoru,
                    'tam_yol': kisi_yolu
                }
                
                # Kişinin resmini bulma
                resim_dosyalari = [f for f in os.listdir(kisi_yolu) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if resim_dosyalari:
                    self.resim_yollari[kisi_id] = os.path.join(kisi_yolu, resim_dosyalari[0])
                    # Resim adını da kişi bilgilerine ekle
                    self.kisi_bilgileri[kisi_id]['resim_adi'] = resim_dosyalari[0]
                
                # Ses dosyalarını toplama
                ses_sayaci = 0
                for dosya in os.listdir(kisi_yolu):
                    if dosya.lower().endswith('.wav'):
                        dosya_yolu = os.path.join(kisi_yolu, dosya)
                        
                        # Dosya boyutu kontrolü
                        dosya_boyutu = os.path.getsize(dosya_yolu)
                        if dosya_boyutu < 100:  # Çok küçük dosyaları atlama
                            logger.warning(f"Çok küçük ses dosyası atlanıyor: {dosya_yolu} ({dosya_boyutu} bayt)")
                            continue
                            
                        # Ses dosyasını yüklemeyi test etme
                        try:
                            waveform, sr = torchaudio.load(dosya_yolu)
                            if waveform.shape[1] < 100:  # Çok kısa sesler
                                logger.warning(f"Çok kısa ses dosyası atlanıyor: {dosya_yolu}")
                                continue
                        except Exception as e:
                            logger.error(f"Ses dosyası yüklenemedi: {dosya_yolu}, Hata: {str(e)}")
                            continue
                            
                        # Dosya adından chunk ID'sini çıkarma
                        try:
                            # "chunk_123.wav" formatındaki dosyalardan ID çıkarma
                            if dosya.startswith("chunk_"):
                                chunk_id = dosya.split('.')[0]  # .wav uzantısını çıkar
                            else:
                                # Eğer başka bir format varsa, dosya adını kullan
                                chunk_id = os.path.splitext(dosya)[0]
                        except:
                            chunk_id = f"unknown_{ses_sayaci}"
                            
                        self.ses_dosyalari.append(dosya_yolu)
                        self.etiketler.append(kisi_id)
                        self.dosya_idleri.append(chunk_id)  # Chunk ID'yi sakla
                        ses_sayaci += 1
                
                logger.info(f"Kişi {kisi_klasoru}: {ses_sayaci} ses dosyası eklendi")
            
            if not self.ses_dosyalari:
                raise ValueError("Hiç geçerli ses dosyası bulunamadı")
                
        except Exception as e:
            logger.error(f"Veri seti oluşturulurken hata: {str(e)}")
            raise
            
        logger.info(f"Toplam {len(self.ses_dosyalari)} ses dosyası, {len(self.resim_yollari)} kişi bulundu.")
    
    def __len__(self):
        return len(self.ses_dosyalari)
    
    @lru_cache(maxsize=128)  # Önbellekleme eklendi
    def ses_dosyasi_yukle(self, ses_yolu):
        """Ses dosyasını yükleyip ön işleme yapar ve önbellekler"""
        try:
            waveform, sample_rate = torchaudio.load(ses_yolu)
            
            # Tek kanala dönüştürme
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Örnekleme hızını ayarlama
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            
            # Sabit uzunluğa getirme
            target_length = self.sample_rate * self.max_duration
            if waveform.shape[1] < target_length:
                waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))
            else:
                waveform = waveform[:, :target_length]
            
            return waveform
            
        except Exception as e:
            logger.error(f"Ses dosyası yüklenirken hata: {ses_yolu}, {str(e)}")
            # Hata durumunda boş bir ses dönelim
            return torch.zeros(1, self.sample_rate * self.max_duration)
    
    def __getitem__(self, idx):
        ses_yolu = self.ses_dosyalari[idx]
        etiket = self.etiketler[idx]
        
        # Ses dosyasını yükleme - önbelleklenmiş
        waveform = self.ses_dosyasi_yukle(ses_yolu)
        
        # MFCC özellik çıkarımı
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80}
        )
        mfccs = mfcc_transform(waveform)
        
        # VERİ ARTIRMA (DATA AUGMENTATION) - transform=True ise
        if self.transform and torch.rand(1).item() < 0.7:  # %70 olasılıkla veri artırma uygula
            # 1. Zaman Maskelemesi (SpecAugment benzeri)
            if torch.rand(1).item() < 0.5:
                time_mask_param = int(mfccs.shape[2] * 0.1)
                time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
                mfccs = time_mask(mfccs)
            
            # 2. Frekans Maskelemesi
            if torch.rand(1).item() < 0.5:
                freq_mask_param = int(mfccs.shape[1] * 0.15)
                freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
                mfccs = freq_mask(mfccs)
            
            # 3. Özellik değerlerine Gaussian gürültü ekleme
            if torch.rand(1).item() < 0.3:
                noise = torch.randn_like(mfccs) * 0.05
                mfccs = mfccs + noise
            
            # 4. Özellik ölçekleme (feature scaling)
            if torch.rand(1).item() < 0.3:
                scale_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.2  # 0.9-1.1 arası
                mfccs = mfccs * scale_factor
        
        return mfccs, etiket
    
    def get_kisi_resmi(self, kisi_id):
        if kisi_id in self.resim_yollari:
            return self.resim_yollari[kisi_id]
        return None

    def get_sinif_sayisi(self):
        """Toplam sınıf (kişi) sayısını döndürür"""
        return max(self.etiketler) + 1
        
    def get_kisi_bilgileri(self):
        """Tüm kişi bilgilerini döndürür"""
        return self.kisi_bilgileri

    def get_dosya_id(self, idx):
        """Belirli bir indeksteki dosyanın ID'sini döndürür"""
        if 0 <= idx < len(self.dosya_idleri):
            return self.dosya_idleri[idx]
        return None

    def get_resim_base64(self, kisi_id):
        """Kişinin resmini base64 formatında kodlar"""
        resim_yolu = self.get_kisi_resmi(kisi_id)
        if resim_yolu and os.path.exists(resim_yolu):
            try:
                with open(resim_yolu, "rb") as f:
                    image_bytes = f.read()
                    return base64.b64encode(image_bytes).decode('utf-8')
            except Exception as e:
                logger.error(f"Resim base64 kodlaması sırasında hata: {str(e)}")
                return None
        return None

# 2. GELİŞMİŞ MODEL MİMARİSİ
class ResidualBlock(nn.Module):
    """Kalıntı (Residual) blok implementasyonu"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.05)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.05)
        
        # Kısayol bağlantısı
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels, momentum=0.05)
            )

    def forward(self, x):
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class FinalModel(nn.Module):
    def __init__(self, num_kisiler=10, dropout_rate=0.5):
        super(FinalModel, self).__init__()
        
        # Giriş konvolüsyon
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.05)
        self.relu = nn.ReLU(inplace=True)
        
        # Kalıntı blokları
        self.residual1 = ResidualBlock(32, 64, stride=1)
        self.residual2 = ResidualBlock(64, 128, stride=1)
        self.residual3 = ResidualBlock(128, 256, stride=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Tam bağlantılı katmanlar
        self.fc1 = nn.Linear(256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_kisiler)
        
        # Modeli ilklendir
        self.initialize_weights()
        
    def initialize_weights(self):
        """Model ağırlıklarını daha iyi başlangıç değerleriyle ilklendirir"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Ara katman aktivasyonlarını saklayacak sözlük
        activations = {}
        
        # Giriş katmanları
        x = self.relu(self.bn1(self.conv1(x)))
        activations['conv1'] = x
        x = self.pool(x)
        
        # Kalıntı blokları
        x = self.residual1(x)
        activations['residual1'] = x
        x = self.pool(x)
        
        x = self.residual2(x)
        activations['residual2'] = x
        x = self.pool(x)
        
        x = self.residual3(x)
        activations['residual3'] = x
        x = self.pool(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        activations['features'] = x
        
        # Tam bağlantılı katmanlar
        x = self.relu(self.bn_fc1(self.fc1(x)))
        activations['fc1'] = x
        x = self.dropout(x)
        
        x = self.fc2(x)
        activations['output'] = x
        
        return x, activations

# 3. KONUŞMACI TANIMA SİSTEMİ SINIFI - GELİŞTİRİLDİ
class KonusmaciTanimaSistemi:
    def __init__(self, veri_klasoru="/home/kemal/Projects/Robotek Latest/veriseti"):
        self.veri_klasoru = veri_klasoru
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.veri_seti = None
        self.train_loader = None
        self.test_loader = None
        self.best_model_state = None
        self.model_metrics = {}
        
        # Çıktı dizinlerini oluştur
        output_dizinleri_olustur()
        
        # Sistem bilgisi
        logger.info(f"Kullanılan cihaz: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"GPU Modeli: {torch.cuda.get_device_name(0)}")
            logger.info(f"Bellek Kullanımı: {torch.cuda.memory_allocated(0)/1e9:.2f} GB / "
                       f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        
        # PyTorch versiyonu
        logger.info(f"PyTorch Versiyonu: {torch.__version__}")
    
    def veri_yukle(self):
        """Veri setini yükle ve eğitim/test olarak ayır"""
        try:
            logger.info(f"Veri seti yükleniyor: {self.veri_klasoru}")
            self.veri_seti = KonusmaciVeriseti(self.veri_klasoru)
            
            # Verileri %90 eğitim, %10 test olarak ayırma
            tum_indeksler = list(range(len(self.veri_seti)))
            etiketler = self.veri_seti.etiketler
            
            train_idx, test_idx = train_test_split(
                tum_indeksler, 
                test_size=0.1,  # %10 test
                stratify=etiketler,  # Her sınıftan eşit oranda örnek
                random_state=42
            )
            
            # Alt kümeler oluşturma
            from torch.utils.data import Subset
            train_set = Subset(self.veri_seti, train_idx)
            test_set = Subset(self.veri_seti, test_idx)
            
            logger.info(f"Eğitim seti: {len(train_set)} örnek")
            logger.info(f"Test seti: {len(test_set)} örnek")
            
            # Test setindeki örneklerin ID'lerini sakla
            self.test_ids = [self.veri_seti.dosya_idleri[i] for i in test_idx]
            
            # Veri yükleyicileri - Prefetch kullanarak performans artışı
            self.train_loader = DataLoader(train_set, batch_size=32, shuffle=True, 
                                           num_workers=4, prefetch_factor=2, 
                                           pin_memory=True)
            self.test_loader = DataLoader(test_set, batch_size=32, shuffle=False, 
                                          num_workers=4, prefetch_factor=2, 
                                          pin_memory=True)
            
            return True
        except Exception as e:
            logger.error(f"Veri yükleme hatası: {str(e)}")
            return False
    
    def model_olustur(self):
        """Model oluştur ve cihaza yükle"""
        try:
            sinif_sayisi = self.veri_seti.get_sinif_sayisi()
            logger.info(f"Model oluşturuluyor, sınıf sayısı: {sinif_sayisi}")
            
            self.model = FinalModel(num_kisiler=sinif_sayisi)
            self.model = self.model.to(self.device)
            
            # Model mimarisini yazdır
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Model Mimarisi:\n{self.model}")
            logger.info(f"Toplam öğrenilebilir parametre sayısı: {total_params:,}")
            
            return True
        except Exception as e:
            logger.error(f"Model oluşturma hatası: {str(e)}")
            return False
    
    def egit(self, num_epochs=20, ogrenme_orani=0.001, batch_size=32):
        """Model eğitimi - GELİŞTİRİLDİ"""
        if self.model is None or self.train_loader is None or self.test_loader is None:
            logger.error("Eğitim başlatılamıyor: Model veya veri yükleyicileri mevcut değil!")
            return False
        
        # CSV log dosyası oluşturma - çıktı klasörünü kullan
        csv_path = os.path.join(LOG_DIR, 'training_metrics.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Training_Loss', 'Training_Accuracy', 'Test_Loss', 'Test_Accuracy', 
                            'Precision', 'Recall', 'F1_Score', 'Macro_F1', 'LR', 'Epoch_Time', 'Total_Time'])
        
        # Optimizer ve kayıp fonksiyonu tanımlama
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=ogrenme_orani, weight_decay=1e-5)
        
        # OneCycleLR - daha kararlı eğitim için
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=ogrenme_orani,
            epochs=num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,  # İlk %30'da artan, geri kalanda düşen lr
            div_factor=10,  # baslangic_lr = max_lr/10
            final_div_factor=100  # min_lr = baslangic_lr/100
        )
        
        # Karma Hassasiyet (Mixed Precision) için scaler
        scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        
        # Eğitim metrikleri
        training_loss = []
        test_acc = []
        best_test_acc = 0.0
        self.best_model_state = None
        
        # Toplam eğitim süresi başlangıcı
        total_start_time = time.time()
        
        try:
            for epoch in range(num_epochs):
                # Epoch başına süre ölçümü
                epoch_start_time = time.time()
                
                # Eğitim modu
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                # İlerleme çubuğunu başlat
                progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
                
                for batch_idx, (inputs, targets) in enumerate(progress_bar):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Gradyanları sıfırlama
                    optimizer.zero_grad()
                    
                    # İleri geçiş - Karma Hassasiyet kullanarak
                    with torch.amp.autocast(device_type=self.device.type):
                        outputs, _ = self.model(inputs)
                        loss = criterion(outputs, targets)
                    
                    # Geri yayılım - Karma Hassasiyet ile
                    scaler.scale(loss).backward()
                    
                    # Gradyan kırpma - kararsızlığı azaltmak için
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Parametre güncelleme - Karma Hassasiyet ile
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Öğrenme oranı güncelleme (OneCycleLR için her batch'te)
                    scheduler.step()
                    
                    # İstatistikler
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    # İlerleme çubuğunu güncelleme
                    progress_bar.set_postfix({
                        'loss': running_loss / (batch_idx + 1),
                        'acc': 100. * correct / total,
                        'lr': scheduler.get_last_lr()[0]
                    })
                
                train_loss = running_loss / len(self.train_loader)
                train_acc = 100. * correct / total
                training_loss.append(train_loss)
                
                # Test modunda değerlendirme
                self.model.eval()
                test_loss = 0.0
                test_correct = 0
                test_total = 0
                
                # Test süresi ölçümü
                test_start_time = time.time()
                
                all_targets = []
                all_predictions = []
                
                with torch.no_grad():
                    for inputs, targets in self.test_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs, _ = self.model(inputs)
                        loss = criterion(outputs, targets)
                        
                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        test_total += targets.size(0)
                        test_correct += predicted.eq(targets).sum().item()
                        
                        # Precision/Recall/F1 için
                        all_targets.extend(targets.cpu().numpy())
                        all_predictions.extend(predicted.cpu().numpy())
                
                test_time = time.time() - test_start_time
                current_test_acc = 100. * test_correct / test_total
                test_acc.append(current_test_acc)
                
                # Precision, Recall, F1 hesaplama
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_targets, all_predictions, average='weighted')
                
                # Macro F1 hesaplama (her sınıf için eşit ağırlık)
                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                    all_targets, all_predictions, average='macro')
                
                # Epoch süresi
                epoch_time = time.time() - epoch_start_time
                total_time = time.time() - total_start_time
                
                # Mevcut öğrenme oranı
                current_lr = scheduler.get_last_lr()[0]
                
                # Detaylı loglama
                epoch_info = (
                    f'Epoch {epoch+1}/{num_epochs}:\n'
                    f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n'
                    f'Test Loss: {test_loss / len(self.test_loader):.4f} | Test Acc: {current_test_acc:.2f}%\n'
                    f'Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Macro F1: {f1_macro:.4f}\n'
                    f'LR: {current_lr:.6f}\n'
                    f'Epoch Süresi: {datetime.timedelta(seconds=int(epoch_time))}\n'
                    f'Toplam Eğitim Süresi: {datetime.timedelta(seconds=int(total_time))}\n'
                    f'Test Süresi: {datetime.timedelta(seconds=int(test_time))}'
                )
                
                logger.info(epoch_info)
                
                # CSV'ye kaydetme
                csv_writer.writerow([
                    epoch+1, train_loss, train_acc, test_loss / len(self.test_loader), current_test_acc,
                    precision, recall, f1, f1_macro, current_lr, epoch_time, total_time
                ])
                csv_file.flush()  # Dosyaya yazma
                
                # En iyi modeli kaydetme
                if current_test_acc > best_test_acc:
                    best_test_acc = current_test_acc
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    
                    # En iyi model dosyasını tarih-saat formatında kaydet
                    tarih_saat = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    best_model_filename = os.path.join(MODEL_DIR, f'final_model_{tarih_saat}.pth')
                    
                    # Resimleri Base64 formatında kodla
                    image_data = {}
                    kisi_bilgileri = self.veri_seti.get_kisi_bilgileri()
                    
                    for kisi_id in kisi_bilgileri.keys():
                        image_base64 = self.veri_seti.get_resim_base64(kisi_id)
                        if image_base64:
                            image_data[kisi_id] = {
                                'base64': image_base64,
                                'kisi_bilgisi': kisi_bilgileri[kisi_id]
                            }
                    
                    # Modeli ve ek bilgileri kaydet
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': train_loss,
                        'test_acc': current_test_acc,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'f1_macro': f1_macro,  # Macro F1'i de kaydet
                        'training_time': total_time,
                        'kisi_resimleri': image_data,
                        'kisi_bilgileri': kisi_bilgileri
                    }, best_model_filename)
                    
                    logger.info(f'En iyi model kaydedildi (Doğruluk: {best_test_acc:.2f}%): {best_model_filename}')
                
                # Keskin doğruluk düşüşlerini tespit et ve handle et
                if epoch > 0 and test_acc[-2] - current_test_acc > 15:  # %15'ten fazla düşüş
                    logger.warning(f"Keskin doğruluk düşüşü tespit edildi! ({test_acc[-2]:.2f}% -> {current_test_acc:.2f}%)")
                    logger.warning(f"Önceki en iyi modele geri dönülüyor...")
                    
                    # En iyi modeli geri yükle
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
            
            # Eğitimin sonunda en iyi modeli geri yükle
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
                logger.info(f"Eğitim tamamlandı. En iyi model yüklendi (Doğruluk: {best_test_acc:.2f}%)")
            
            # CSV dosyasını kapatma
            csv_file.close()
            
            # Toplam eğitim süresi
            total_training_time = time.time() - total_start_time
            logger.info(f'Toplam eğitim süresi: {datetime.timedelta(seconds=int(total_training_time))}')
            
            # Eğitim istatistiklerini çizme
            self.egitim_istatistiklerini_ciz(training_loss, test_acc)
            
            # Eğitim metriklerini kaydetme
            self.model_metrics = {
                'accuracy': best_test_acc,
                'training_time': total_training_time,
                'epochs': num_epochs,
                'f1_macro': f1_macro
            }
            
            # Eğitim tamamlandıktan sonra modeli son haliyle de kaydet
            self.modeli_kaydet()
            
            # JSON formatında test tahminlerini çıkart
            self.test_tahmin_json_olustur()
            
            return True
            
        except Exception as e:
            logger.error(f"Eğitim sırasında hata: {str(e)}", exc_info=True)
            csv_file.close()
            return False
    
    def egitim_istatistiklerini_ciz(self, training_loss, test_acc):
        """Eğitim istatistiklerini görselleştirir"""
        try:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(training_loss)
            plt.title('Eğitim Kaybı')
            plt.xlabel('Epoch')
            plt.ylabel('Kayıp')
            
            plt.subplot(2, 2, 2)
            plt.plot(test_acc)
            plt.title('Test Doğruluğu')
            plt.xlabel('Epoch')
            plt.ylabel('Doğruluk (%)')
            
            # Kayıp ve doğruluğu aynı grafikte göster
            plt.subplot(2, 1, 2)
            epochs = range(1, len(training_loss) + 1)
            plt.plot(epochs, training_loss, 'b-', label='Eğitim Kaybı')
            plt.plot(epochs, [l/100 for l in test_acc], 'r-', label='Test Doğruluğu')  # ölçek ayarlama
            plt.title('Eğitim Kaybı ve Test Doğruluğu')
            plt.xlabel('Epoch')
            plt.legend()
            
            plt.tight_layout()
            
            # Grafiği tarih-saat formatında kaydet - çıktı klasörünü kullan
            tarih_saat = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            plt.savefig(os.path.join(GRAPH_DIR, f'egitim_istatistikleri_{tarih_saat}.png'))
            plt.close()  # Açık tutmamak için kapat
            logger.info(f"Eğitim istatistikleri görselleştirildi ve kaydedildi: {os.path.join(GRAPH_DIR, f'egitim_istatistikleri_{tarih_saat}.png')}")
        except Exception as e:
            logger.error(f"Eğitim istatistiklerini görselleştirme hatası: {str(e)}")
    
    def test_et(self):
        """Model test performansını ölçer"""
        if self.model is None or self.test_loader is None:
            logger.error("Test başlatılamıyor: Model veya test veri yükleyicisi mevcut değil!")
            return False
            
        logger.info("Test işlemi başlıyor...")
        test_start_time = time.time()
        
        # Test modu
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        correct = 0
        total = 0
        
        all_targets = []
        all_predictions = []
        class_correct = [0] * self.veri_seti.get_sinif_sayisi()
        class_total = [0] * self.veri_seti.get_sinif_sayisi()
        
        # Tüm örneklerin tahmin süresi
        ornek_sureler = []
        
        try:
            with torch.no_grad():
                for inputs, targets in tqdm(self.test_loader, desc="Test Ediliyor"):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Örnekler için tahmin süresi
                    start_time = time.time()
                    outputs, _ = self.model(inputs)
                    ornek_sure = time.time() - start_time
                    ornek_sureler.append(ornek_sure)
                    
                    loss = criterion(outputs, targets)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    # Sınıf bazlı doğruluk
                    for i in range(targets.size(0)):
                        label = targets[i].item()
                        prediction = predicted[i].item()
                        if label == prediction:
                            class_correct[label] += 1
                        class_total[label] += 1
                    
                    # Metrikler için
                    all_targets.extend(targets.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
            
            # Test metrikleri
            accuracy = 100. * correct / total
            avg_loss = test_loss / len(self.test_loader)
            
            # Sınıf bazlı doğruluk raporu
            logger.info("Sınıf Bazlı Doğruluk:")
            sinif_dogruluk = []
            for i in range(self.veri_seti.get_sinif_sayisi()):
                if class_total[i] > 0:
                    class_acc = 100 * class_correct[i] / class_total[i]
                    sinif_dogruluk.append(class_acc)
                    logger.info(f'Person{i+1}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})')
                else:
                    sinif_dogruluk.append(0)
            
            # Tahmin süresi istatistikleri
            avg_tahmin_suresi = sum(ornek_sureler) / len(ornek_sureler)
            batch_size = self.test_loader.batch_size
            ornekbasi_tahmin_suresi = avg_tahmin_suresi / batch_size
            
            # Precision, Recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_predictions, average='weighted')
                
            # Macro F1 hesaplama
            _, _, f1_macro, _ = precision_recall_fscore_support(
                all_targets, all_predictions, average='macro')
            
            # Toplam test süresi
            toplam_test_suresi = time.time() - test_start_time
            
            # Test sonuçlarını loglama
            test_sonuclari = (
                f"Test Sonuçları:\n"
                f"Doğruluk: {accuracy:.2f}%\n"
                f"Kayıp: {avg_loss:.4f}\n"
                f"Precision: {precision:.4f}\n"
                f"Recall: {recall:.4f}\n"
                f"F1 Score: {f1:.4f}\n"
                f"Macro F1 Score: {f1_macro:.4f}\n"
                f"Toplam Test Süresi: {datetime.timedelta(seconds=int(toplam_test_suresi))}\n"
                f"Ortalama Batch Tahmin Süresi: {avg_tahmin_suresi:.4f} saniye\n"
                f"Örnek Başına Tahmin Süresi: {ornekbasi_tahmin_suresi*1000:.2f} milisaniye"
            )
            
            logger.info(test_sonuclari)
            
            # Tarih-saat formatı
            tarih_saat = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            
            # Karışıklık matrisi
            self.karisiklik_matrisi_ciz(all_targets, all_predictions, tarih_saat)
            
            # Sınıflandırma raporu
            class_report = classification_report(all_targets, all_predictions, 
                                                target_names=[f'Person{i+1}' for i in range(self.veri_seti.get_sinif_sayisi())])
            logger.info(f"Sınıflandırma Raporu:\n{class_report}")
            
            # Sonuçları dosyaya da kaydet
            with open(os.path.join(LOG_DIR, f'test_sonuclari_{tarih_saat}.txt'), 'w') as f:
                f.write(test_sonuclari)
                f.write("\n\nSınıflandırma Raporu:\n")
                f.write(class_report)
            
            # Test metriklerini kaydet
            self.model_metrics.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'f1_macro': f1_macro,
                'test_suresi': toplam_test_suresi,
                'ortalama_tahmin_suresi': avg_tahmin_suresi,
                'ornekbasi_tahmin_suresi': ornekbasi_tahmin_suresi,
                'class_accuracy': sinif_dogruluk,
                'test_timestamp': tarih_saat
            })
            
            # JSON formatında test tahminlerini çıkart
            self.test_tahmin_json_olustur()
            
            return self.model_metrics
            
        except Exception as e:
            logger.error(f"Test sırasında hata: {str(e)}", exc_info=True)
            return False
    
    def karisiklik_matrisi_ciz(self, all_targets, all_predictions, timestamp=None):
        """Karışıklık matrisini görselleştirir"""
        try:
            cm = confusion_matrix(all_targets, all_predictions)
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=[f'Person{i+1}' for i in range(self.veri_seti.get_sinif_sayisi())],
                        yticklabels=[f'Person{i+1}' for i in range(self.veri_seti.get_sinif_sayisi())])
            plt.xlabel('Tahmin Edilen')
            plt.ylabel('Gerçek')
            plt.title('Karışıklık Matrisi')
            
            # Zaman damgası eklenerek kaydet
            if timestamp is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                
            plt.savefig(os.path.join(GRAPH_DIR, f'karisiklik_matrisi_{timestamp}.png'))
            plt.close()  # Açık tutmamak için kapat
            logger.info(f"Karışıklık matrisi görselleştirildi ve kaydedildi: {os.path.join(GRAPH_DIR, f'karisiklik_matrisi_{timestamp}.png')}")
        except Exception as e:
            logger.error(f"Karışıklık matrisi çizim hatası: {str(e)}")
    
    def modeli_kaydet(self, dosya_adi=None):
        """Modeli diske kaydeder"""
        if self.model is None:
            logger.error("Model kaydedilemedi: Model mevcut değil!")
            return False
            
        try:
            # Tarih-saat formatında dosya adı oluştur
            if dosya_adi is None:
                tarih_saat = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                dosya_adi = os.path.join(MODEL_DIR, f"final_model_{tarih_saat}.pth")
            
            # Kişi resimlerini Base64 formatında kodla
            image_data = {}
            kisi_bilgileri = self.veri_seti.get_kisi_bilgileri()
            
            for kisi_id in kisi_bilgileri.keys():
                image_base64 = self.veri_seti.get_resim_base64(kisi_id)
                if image_base64:
                    image_data[kisi_id] = {
                        'base64': image_base64,
                        'kisi_bilgisi': kisi_bilgileri[kisi_id]
                    }
            
            # Modeli ve ek bilgileri kaydet
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_metrics': self.model_metrics,
                'kisi_resimleri': image_data,
                'kisi_bilgileri': kisi_bilgileri,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, dosya_adi)
            
            logger.info(f"Model başarıyla kaydedildi: {dosya_adi}")
            return True
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {str(e)}")
            return False
    
    def modeli_yukle(self, dosya_adi):
        """Kaydedilmiş modeli yükler"""
        try:
            # Önce sınıf sayısını al ve modeli oluştur
            if self.veri_seti is None:
                self.veri_yukle()
                
            if self.model is None:
                self.model_olustur()
                
            # Model yolu kontrolü
            if not os.path.isabs(dosya_adi):
                dosya_adi = os.path.join(MODEL_DIR, dosya_adi)
                
            # Model ağırlıklarını yükle
            checkpoint = torch.load(dosya_adi, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model_metrics = checkpoint.get('model_metrics', {})
            
            # Kişi bilgilerini yükle
            kisi_resimleri = checkpoint.get('kisi_resimleri', {})
            logger.info(f"Model başarıyla yüklendi: {dosya_adi}")
            logger.info(f"Model metrikleri: {self.model_metrics}")
            logger.info(f"Yüklenen kişi resmi sayısı: {len(kisi_resimleri)}")
            return True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}", exc_info=True)
            return False
    
    def ses_tahmini(self, ses_dosyasi):
        """Ses dosyasından konuşmacı tahmini yapar ve sonuçları döndürür"""
        if self.model is None:
            logger.error("Tahmin yapılamıyor: Model mevcut değil!")
            return None
            
        try:
            # Dosya kontrolü
            if not os.path.exists(ses_dosyasi):
                raise FileNotFoundError(f"Ses dosyası bulunamadı: {ses_dosyasi}")
                
            if not ses_dosyasi.lower().endswith('.wav'):
                raise ValueError("Sadece WAV formatında dosyalar desteklenir")
                
            # Dosya boyutu kontrolü
            if os.path.getsize(ses_dosyasi) > 10 * 1024 * 1024:  # 10 MB
                raise ValueError("Dosya boyutu çok büyük")
                
            # Tahmin süresi ölçümü başlangıcı
            tahmin_baslangic = time.time()
            self.model.eval()
            
            # Ses dosyasını yükleme
            yukleme_baslangic = time.time()
            waveform, sample_rate = torchaudio.load(ses_dosyasi)
            yukleme_suresi = time.time() - yukleme_baslangic
            
            # Tek kanala dönüştürme
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Örnekleme hızını ayarlama
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            
            # Sabit uzunluğa getirme
            target_length = 16000 * 5  # 5 saniye
            if waveform.shape[1] < target_length:
                waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))
            else:
                waveform = waveform[:, :target_length]
            
            # MFCC özellik çıkarımı
            onisleme_baslangic = time.time()
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=16000,
                n_mfcc=40,
                melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80}
            )
            mfccs = mfcc_transform(waveform)
            onisleme_suresi = time.time() - onisleme_baslangic
            
            # Tahmin için hazırlama
            mfccs = mfccs.unsqueeze(0).to(self.device)  # Batch boyutu ekleme
            
            # Tahmin
            model_tahmin_baslangic = time.time()
            with torch.no_grad():
                outputs, aktivasyonlar = self.model(mfccs)
                _, predicted = outputs.max(1)
                kisi_id = predicted.item()
            model_tahmin_suresi = time.time() - model_tahmin_baslangic
            
            # Tahmin olasılıkları
            probabilities = F.softmax(outputs, dim=1)[0] * 100
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            # Toplam tahmin süresi
            toplam_tahmin_suresi = time.time() - tahmin_baslangic
            
            # Dosya adından chunk ID'yi çıkar
            try:
                chunk_id = os.path.splitext(os.path.basename(ses_dosyasi))[0]
            except:
                chunk_id = f"unknown_{int(time.time())}"
            
            # Sonuçları loglama
            tahmin_raporu = (
                f"Tahmin edilen kişi: Person{kisi_id+1} (ID: {chunk_id})\n"
                f"Yükleme süresi: {yukleme_suresi*1000:.2f} ms\n"
                f"Önişleme süresi: {onisleme_suresi*1000:.2f} ms\n"
                f"Model tahmin süresi: {model_tahmin_suresi*1000:.2f} ms\n"
                f"Toplam tahmin süresi: {toplam_tahmin_suresi*1000:.2f} ms"
            )
            logger.info(tahmin_raporu)
            
            # Tahmin olasılıklarını loglama
            logger.info("Tahmin olasılıkları:")
            for i, (prob, idx) in enumerate(zip(top_probs.cpu().numpy(), top_indices.cpu().numpy())):
                logger.info(f"  {i+1}. Person{idx+1}: {prob:.2f}%")
            
            # Sonuçları döndür
            return {
                'kisi_id': kisi_id,
                'probabilities': probabilities.cpu().numpy(),
                'top_indices': top_indices.cpu().numpy(),
                'top_probs': top_probs.cpu().numpy(),
                'waveform': waveform.cpu().numpy(),
                'mfccs': mfccs[0].cpu().numpy(),
                'aktivasyonlar': {k: v[0].cpu().detach().numpy() if isinstance(v, torch.Tensor) else v 
                                  for k, v in aktivasyonlar.items()},
                'tahmin_suresi': toplam_tahmin_suresi,
                'yukleme_suresi': yukleme_suresi,
                'onisleme_suresi': onisleme_suresi,
                'model_tahmin_suresi': model_tahmin_suresi,
                'chunk_id': chunk_id
            }
            
        except Exception as e:
            logger.error(f"Tahmin sırasında hata: {str(e)}", exc_info=True)
            return None
    
    def tahmin_ve_gorsellestir(self, ses_dosyasi):
        """Ses tahmini yapar ve görselleştirir"""
        try:
            # Önce tahmini yap
            tahmin_sonucu = self.ses_tahmini(ses_dosyasi)
            if tahmin_sonucu is None:
                return None
                
            # Sonuçları çıkar
            kisi_id = tahmin_sonucu['kisi_id']
            waveform = tahmin_sonucu['waveform']
            top_indices = tahmin_sonucu['top_indices']
            top_probs = tahmin_sonucu['top_probs']
            toplam_tahmin_suresi = tahmin_sonucu['tahmin_suresi']
            chunk_id = tahmin_sonucu['chunk_id']
            
            # Zaman damgası
            tarih_saat = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            
            # Görselleştirme
            plt.figure(figsize=(15, 12))
            
            # Ses dalgası
            plt.subplot(3, 1, 1)
            plt.title(f"Ses Dalgası - {chunk_id}")
            plt.plot(waveform[0])
            plt.grid(True)
            
            # Spektrogram
            plt.subplot(3, 1, 2)
            plt.title("Mel Spektrogram")
            melspec = torchaudio.transforms.MelSpectrogram()(torch.from_numpy(waveform))
            melspec_db = 20 * torch.log10(melspec + 1e-9)
            plt.imshow(melspec_db[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            
            # Kişinin resmini gösterme ve tahmin sonuçları
            plt.subplot(3, 1, 3)
            resim_yolu = self.veri_seti.get_kisi_resmi(kisi_id)
            
            # Tahmin çubuk grafiği için alt grafik oluştur
            ax1 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
            
            # En olası 3 kişi için olasılık çubukları
            bars = ax1.bar(
                [f'Person{idx+1}' for idx in top_indices],
                top_probs,
                color=['green', 'lightgreen', 'lightblue']
            )
            
            # Değerleri çubukların üzerine ekle
            for bar, prob in zip(bars, top_probs):
                ax1.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f'{prob:.1f}%',
                    ha='center'
                )
            
            ax1.set_ylim([0, 100])
            ax1.set_title(f"Tahmin Olasılıkları (Toplam Süre: {toplam_tahmin_suresi*1000:.1f} ms)")
            ax1.set_ylabel('Olasılık (%)')
            
            # Kişi resmi için alt grafik
            ax2 = plt.subplot2grid((3, 3), (2, 2))
            if resim_yolu:
                img = Image.open(resim_yolu)
                ax2.imshow(np.array(img))
                ax2.set_title(f"Konuşmacı: Person{kisi_id+1}")
                ax2.axis('off')
            else:
                ax2.text(0.5, 0.5, f"Person{kisi_id+1} için resim bulunamadı", 
                        horizontalalignment='center', verticalalignment='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(GRAPH_DIR, f'tahmin_sonucu_{chunk_id}_{tarih_saat}.png'), dpi=300)
            plt.close()  # Açık tutmamak için kapat
            
            # Model aktivasyonlarını görselleştir
            self.ara_katman_aktivasyonlarini_goster(tahmin_sonucu['aktivasyonlar'], f'{chunk_id}_{tarih_saat}')
            
            # Tahmin sonucunu JSON olarak kaydet
            tahmin_json = {
                "yarisma": YARISMA_ADI,
                "takim_adi": TAKIM_ADI,
                "takim_id": TAKIM_ID,
                "tahminler": [
                    {
                        "id": chunk_id,
                        "tahmin_etiketi": f"person{kisi_id+1}"
                    }
                ]
            }
            
            json_path = os.path.join(PREDICTION_DIR, f'tahmin_{chunk_id}_{tarih_saat}.json')
            with open(json_path, 'w') as f:
                json.dump(tahmin_json, f, indent=2)
                
            logger.info(f"Tahmin JSON dosyası oluşturuldu: {json_path}")
            
            return tahmin_sonucu
        except Exception as e:
            logger.error(f"Görselleştirme hatası: {str(e)}", exc_info=True)
            return None
    
    def ara_katman_aktivasyonlarini_goster(self, aktivasyonlar, timestamp=None):
        """Model ara katman aktivasyonlarını görselleştirir"""
        try:
            plt.figure(figsize=(15, 12))
            plt.suptitle("Model Ara Katman Aktivasyonları", fontsize=16)
            
            # Katmanları görselleştir
            plot_index = 1
            
            for layer_name, activation in aktivasyonlar.items():
                if isinstance(activation, np.ndarray):
                    # Aktivasyon tensör boyutunu kontrol et ve uygun şekilde işle
                    if len(activation.shape) == 3 and plot_index <= 6:  # Konvolüsyonel katmanlar, en fazla 6 tane göster
                        plt.subplot(3, 2, plot_index)
                        plt.title(f"Katman: {layer_name}")
                        
                        # İlk kanal
                        if activation.shape[0] > 0:
                            plt.imshow(activation[0], aspect='auto', cmap='viridis')
                            plt.colorbar(format='%.2f')
                        plot_index += 1
                    elif len(activation.shape) == 1 and plot_index <= 6:  # Tam bağlantılı katmanlar
                        plt.subplot(3, 2, plot_index)
                        plt.title(f"Katman: {layer_name}")
                        plt.bar(range(min(50, len(activation))), activation[:50])  # İlk 50 nöronu göster
                        plt.grid(True)
                        plot_index += 1
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # suptitle için yer bırak
            
            if timestamp is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            
            plt.savefig(os.path.join(GRAPH_DIR, f'model_aktivasyonlari_{timestamp}.png'), dpi=300)
            plt.close()  # Açık tutmamak için kapat
            logger.info(f"Model aktivasyonları görselleştirildi ve kaydedildi: {os.path.join(GRAPH_DIR, f'model_aktivasyonlari_{timestamp}.png')}")
            
            return True
        except Exception as e:
            logger.error(f"Aktivasyon görselleştirme hatası: {str(e)}", exc_info=True)
            return False
    
    def test_tahmin_json_olustur(self):
        """Test seti için yarışma formatında JSON çıktısı oluşturur"""
        if self.model is None or self.test_loader is None:
            logger.error("JSON çıktısı oluşturulamıyor: Model veya test veri yükleyicisi mevcut değil!")
            return False
            
        try:
            logger.info("Test seti için JSON tahmin çıktısı oluşturuluyor...")
            self.model.eval()
            
            all_predictions = []
            test_ids = []
            
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(self.test_loader):
                    inputs = inputs.to(self.device)
                    outputs, _ = self.model(inputs)
                    _, predicted = outputs.max(1)
                    
                    # Test set subset indekslerini orijinal veri seti indekslerine dönüştür
                    for j, pred in enumerate(predicted):
                        # Test loader'dan indeksi al
                        idx = i * self.test_loader.batch_size + j
                        if idx < len(self.test_loader.dataset):
                            # Subset indeksini orijinal indekse dönüştür
                            orig_idx = self.test_loader.dataset.indices[idx]
                            # Chunk ID'yi al
                            chunk_id = self.veri_seti.get_dosya_id(orig_idx)
                            kisi_id = pred.item()
                            
                            all_predictions.append({
                                "id": chunk_id,
                                "tahmin_etiketi": f"person{kisi_id+1}"
                            })
                            test_ids.append(chunk_id)
            
            # JSON çıktısını oluştur
            json_data = {
                "yarisma": YARISMA_ADI,
                "takim_adi": TAKIM_ADI,
                "takim_id": TAKIM_ID,
                "tahminler": all_predictions
            }
            
            # JSON dosyasını kaydet
            tarih_saat = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            json_path = os.path.join(PREDICTION_DIR, f'test_tahminleri_{tarih_saat}.json')
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
                
            logger.info(f"Test tahminleri JSON dosyası oluşturuldu: {json_path}")
            logger.info(f"Toplam {len(all_predictions)} test tahmini oluşturuldu")
            
            return True
        except Exception as e:
            logger.error(f"JSON çıktısı oluşturma hatası: {str(e)}", exc_info=True)
            return False
    
    def toplu_tahmin_yap(self, ses_klasoru):
        """Bir klasördeki tüm ses dosyaları için toplu tahmin yapar ve JSON çıktısı oluşturur"""
        if self.model is None:
            logger.error("Toplu tahmin yapılamıyor: Model mevcut değil!")
            return False
            
        try:
            if not os.path.exists(ses_klasoru):
                raise FileNotFoundError(f"Ses klasörü bulunamadı: {ses_klasoru}")
                
            logger.info(f"Toplu tahmin başlatılıyor: {ses_klasoru}")
            
            # Ses dosyalarını bul
            ses_dosyalari = [os.path.join(ses_klasoru, f) for f in os.listdir(ses_klasoru) 
                            if f.lower().endswith('.wav')]
            
            if not ses_dosyalari:
                logger.warning(f"Klasörde ses dosyası bulunamadı: {ses_klasoru}")
                return False
                
            logger.info(f"Toplam {len(ses_dosyalari)} ses dosyası bulundu")
            
            # Tüm tahminleri topla
            all_predictions = []
            
            for ses_dosyasi in tqdm(ses_dosyalari, desc="Tahmin ediliyor"):
                tahmin = self.ses_tahmini(ses_dosyasi)
                if tahmin:
                    all_predictions.append({
                        "id": tahmin['chunk_id'],
                        "tahmin_etiketi": f"person{tahmin['kisi_id']+1}"
                    })
            
            # JSON çıktısını oluştur
            json_data = {
                "yarisma": YARISMA_ADI,
                "takim_adi": TAKIM_ADI,
                "takim_id": TAKIM_ID,
                "tahminler": all_predictions
            }
            
            # JSON dosyasını kaydet
            tarih_saat = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            json_path = os.path.join(PREDICTION_DIR, f'toplu_tahminler_{tarih_saat}.json')
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
                
            logger.info(f"Toplu tahminler JSON dosyası oluşturuldu: {json_path}")
            logger.info(f"Toplam {len(all_predictions)} tahmin oluşturuldu")
            
            return json_path
        except Exception as e:
            logger.error(f"Toplu tahmin hatası: {str(e)}", exc_info=True)
            return False

def main():
    """Ana program"""
    # Çıktı dizinleri oluştur
    output_dizinleri_olustur()
    
    # PyTorch ayarları
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    
    # Program başlangıç zamanı
    program_baslangic = time.time()
    
    # Veri klasörü
    veri_klasoru = "/home/kemal/Projects/Robotek Latest/veriseti"
    
    # Konuşmacı tanıma sistemi
    sistem = KonusmaciTanimaSistemi(veri_klasoru)
    
    # Veri yükleme
    logger.info("1. Veri yükleniyor...")
    if not sistem.veri_yukle():
        logger.error("Veri yüklenemedi! Program sonlandırılıyor.")
        return
    
    # Model oluşturma
    logger.info("2. Model oluşturuluyor...")
    if not sistem.model_olustur():
        logger.error("Model oluşturulamadı! Program sonlandırılıyor.")
        return
    
    # Model eğitimi
    logger.info("3. Model eğitiliyor...")
    if not sistem.egit(num_epochs=20, ogrenme_orani=0.001, batch_size=32):
        logger.error("Model eğitimi başarısız! Program devam ediyor...")
    
    # Model testi
    logger.info("4. Model test ediliyor...")
    test_sonuclari = sistem.test_et()
    
    if test_sonuclari:
        logger.info(f"Test sonuçları: Doğruluk: {test_sonuclari['accuracy']:.2f}%")
    else:
        logger.error("Model testi başarısız!")
    
    # Model kaydetme - tarih-saat formatlı dosya adı
    tarih_saat = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info(f"5. Model kaydediliyor... (final_model_{tarih_saat}.pth)")
    model_dosyasi = os.path.join(MODEL_DIR, f"final_model_{tarih_saat}.pth")
    sistem.modeli_kaydet(model_dosyasi)
    
    # Örnek tahmin
    logger.info("6. Örnek tahmin yapılıyor...")
    # Veri setinden rastgele bir ses dosyası seç
    if sistem.veri_seti and sistem.veri_seti.ses_dosyalari:
        import random
        ornek_ses = random.choice(sistem.veri_seti.ses_dosyalari)
        logger.info(f"Rastgele seçilen ses dosyası: {ornek_ses}")
        
        tahmin_sonucu = sistem.tahmin_ve_gorsellestir(ornek_ses)
        if tahmin_sonucu:
            logger.info(f"Tahmin sonucu: Person{tahmin_sonucu['kisi_id']+1}")
    
    # Program süresi
    toplam_sure = time.time() - program_baslangic
    logger.info(f"Toplam program süresi: {datetime.timedelta(seconds=int(toplam_sure))}")
    
    # Performans özeti
    performans_ozeti = {
        'toplam_program_suresi': toplam_sure,
        'egitim_suresi': sistem.model_metrics.get('training_time', 0),
        'accuracy': sistem.model_metrics.get('accuracy', 0),
        'precision': sistem.model_metrics.get('precision', 0),
        'recall': sistem.model_metrics.get('recall', 0),
        'f1': sistem.model_metrics.get('f1', 0),
        'f1_macro': sistem.model_metrics.get('f1_macro', 0),
        'tahmin_suresi': sistem.model_metrics.get('ornekbasi_tahmin_suresi', 0) * 1000,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Performans özetini yazdır
    logger.info("7. Performans özeti:")
    logger.info(f"Toplam program süresi: {datetime.timedelta(seconds=int(performans_ozeti['toplam_program_suresi']))}")
    logger.info(f"Eğitim süresi: {datetime.timedelta(seconds=int(performans_ozeti['egitim_suresi']))}")
    logger.info(f"Model doğruluğu: {performans_ozeti['accuracy']:.2f}%")
    logger.info(f"Precision: {performans_ozeti['precision']:.4f}")
    logger.info(f"Recall: {performans_ozeti['recall']:.4f}")
    logger.info(f"F1 Score: {performans_ozeti['f1']:.4f}")
    logger.info(f"Macro F1 Score: {performans_ozeti['f1_macro']:.4f}")
    logger.info(f"Örnek başına tahmin süresi: {performans_ozeti['tahmin_suresi']:.2f} ms")
    
    # JSON olarak kaydet
    with open(os.path.join(LOG_DIR, f'performans_ozeti_{tarih_saat}.json'), 'w') as f:
        json.dump(performans_ozeti, f, indent=4)
    logger.info(f"Performans özeti '{os.path.join(LOG_DIR, f'performans_ozeti_{tarih_saat}.json')}' dosyasına kaydedildi.")
    
    # Yarışma JSON çıktısını oluştur
    sistem.test_tahmin_json_olustur()
    
    logger.info("Program başarıyla tamamlandı!")

if __name__ == "__main__":
    main()