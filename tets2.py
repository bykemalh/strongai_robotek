import sys
import os
import json
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from functools import partial, lru_cache
from PIL import Image
import cv2

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QPushButton, QLabel, QFileDialog, QTextEdit, QGroupBox, 
                              QProgressBar, QTableWidget, QTableWidgetItem, 
                              QHeaderView, QMessageBox, QSplitter)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont, QPixmap, QImageReader

# YARISMA BİLGİLERİ
YARISMA_ADI = "robotek"
TAKIM_ADI = "StrongAI"
TAKIM_ID = "team_56"

#==================== MODEL SINIFI ====================
class ResidualBlock(torch.nn.Module):
    """Kalıntı (Residual) blok implementasyonu"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels, momentum=0.05)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels, momentum=0.05)
        
        # Kısayol bağlantısı
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(out_channels, momentum=0.05)
            )

    def forward(self, x):
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class FinalModel(torch.nn.Module):
    def __init__(self, num_kisiler=10, dropout_rate=0.5):
        super(FinalModel, self).__init__()
        
        # Giriş konvolüsyon
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32, momentum=0.05)
        self.relu = torch.nn.ReLU(inplace=True)
        
        # Kalıntı blokları
        self.residual1 = ResidualBlock(32, 64, stride=1)
        self.residual2 = ResidualBlock(64, 128, stride=1)
        self.residual3 = ResidualBlock(128, 256, stride=1)
        
        # Pooling
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        # Tam bağlantılı katmanlar
        self.fc1 = torch.nn.Linear(256, 512)
        self.bn_fc1 = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(512, num_kisiler)
        
        # Modeli ilklendir
        self.initialize_weights()
        
    def initialize_weights(self):
        """Model ağırlıklarını daha iyi başlangıç değerleriyle ilklendirir"""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)
        
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

#==================== TAHMİN SİSTEMİ SINIFI ====================
class KonusmaciTanimaSistemi:
    def __init__(self):
        # GPU varsa kullan, yoksa CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_count = 0
        
        print(f"Cihaz: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def modeli_yukle(self, model_path):
        """Kaydedilmiş modeli yükler"""
        try:
            # Model yolu kontrolü
            if not os.path.exists(model_path):
                print(f"Hata: Model dosyası bulunamadı: {model_path}")
                return False
            
            # Model dosyasını yükle
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Sınıf sayısını belirle
            kisi_bilgileri = checkpoint.get('kisi_bilgileri', {})
            self.class_count = len(kisi_bilgileri) if kisi_bilgileri else 10  # Varsayılan 10
            
            # Model oluştur
            self.model = FinalModel(num_kisiler=self.class_count)
            self.model = self.model.to(self.device)
            
            # Model ağırlıklarını yükle
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Değerlendirme moduna al
            self.model.eval()
            
            print(f"Model başarıyla yüklendi: {model_path}")
            print(f"Toplam sınıf sayısı: {self.class_count}")
            
            return True
            
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            return False
    
    @lru_cache(maxsize=32)
    def ses_dosyasini_isle(self, ses_yolu, sample_rate=16000, max_duration=5):
        """Ses dosyasını önişlemeye tabi tutar - çoklu format desteği"""
        try:
            # Ses dosyasını yükle (torchaudio çoğu formatı destekler)
            waveform, sample_rate_orig = torchaudio.load(ses_yolu)
            
            # Tek kanala dönüştürme
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Örnekleme hızını ayarlama
            if sample_rate_orig != sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate_orig, new_freq=sample_rate
                )
                waveform = resampler(waveform)
            
            # Sabit uzunluğa getirme
            target_length = sample_rate * max_duration
            if waveform.shape[1] < target_length:
                waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
            else:
                waveform = waveform[:, :target_length]
            
            return waveform
            
        except Exception as e:
            print(f"Ses dosyası yükleme hatası: {str(e)}")
            return torch.zeros(1, sample_rate * max_duration)
    
    def ses_tahmini(self, ses_dosyasi):
        """Ses dosyasından konuşmacı tahmini yapar"""
        if self.model is None:
            print("Hata: Model yüklenmemiş!")
            return None
            
        try:
            # Dosya kontrolü
            if not os.path.exists(ses_dosyasi):
                print(f"Hata: Ses dosyası bulunamadı: {ses_dosyasi}")
                return None
                
            # Desteklenen formatlar
            supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma']
            if not any(ses_dosyasi.lower().endswith(fmt) for fmt in supported_formats):
                print(f"Hata: Desteklenmeyen format. Desteklenen formatlar: {', '.join(supported_formats)}")
                return None
                
            # Tahmin başlangıcı
            self.model.eval()
            
            # Ses dosyasını yükle ve işle
            waveform = self.ses_dosyasini_isle(ses_dosyasi)
            
            # MFCC özellik çıkarımı
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=16000,
                n_mfcc=40,
                melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80}
            )
            mfccs = mfcc_transform(waveform)
            
            # Tahmin için hazırlama
            mfccs = mfccs.unsqueeze(0).to(self.device)  # Batch boyutu ekleme
            
            # Tahmin
            with torch.no_grad():
                outputs, _ = self.model(mfccs)
                _, predicted = outputs.max(1)
                kisi_id = predicted.item()
            
            # Tahmin olasılıkları
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
            
            # Dosya adından ID'yi çıkar
            try:
                chunk_id = os.path.splitext(os.path.basename(ses_dosyasi))[0]
            except:
                chunk_id = f"unknown_{int(datetime.now().timestamp())}"
            
            # Sonuçları döndür
            return {
                'kisi_id': kisi_id,
                'chunk_id': chunk_id,
                'probabilities': probabilities.cpu().numpy(),
            }
            
        except Exception as e:
            print(f"Tahmin sırasında hata: {str(e)}")
            return None

#==================== TAHMIN THREAD SINIFI ====================
class TahminThread(QThread):
    progress = Signal(int)
    finished = Signal(list)
    error = Signal(str)
    
    def __init__(self, sistem, klasor_veya_dosya, is_folder=True):
        super().__init__()
        self.sistem = sistem
        self.klasor_veya_dosya = klasor_veya_dosya
        self.is_folder = is_folder
        
    def run(self):
        try:
            all_predictions = []
            
            # Desteklenen ses formatları
            supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma']
            
            if self.is_folder:
                # Klasördeki tüm ses dosyalarını bul
                files = []
                for file in os.listdir(self.klasor_veya_dosya):
                    if any(file.lower().endswith(fmt) for fmt in supported_formats):
                        files.append(os.path.join(self.klasor_veya_dosya, file))
                
                # Dosya yoksa hata ver
                if not files:
                    self.error.emit("Klasörde desteklenen ses dosyası bulunamadı!")
                    return
                
                # Her dosya için tahmin yap
                for i, file in enumerate(files):
                    tahmin = self.sistem.ses_tahmini(file)
                    if tahmin:
                        all_predictions.append({
                            "id": tahmin['chunk_id'],
                            "tahmin_etiketi": f"person{tahmin['kisi_id']+1}"
                        })
                    # İlerlemeyi bildir
                    self.progress.emit(int((i+1) / len(files) * 100))
            else:
                # Tek dosya için tahmin yap
                tahmin = self.sistem.ses_tahmini(self.klasor_veya_dosya)
                if tahmin:
                    all_predictions.append({
                        "id": tahmin['chunk_id'],
                        "tahmin_etiketi": f"person{tahmin['kisi_id']+1}"
                    })
                self.progress.emit(100)
            
            # Sonuçları bildir
            self.finished.emit(all_predictions)
            
        except Exception as e:
            self.error.emit(f"Tahmin sırasında hata: {str(e)}")

#==================== ANA UYGULAMA SINIFI ====================
class FinalModelTester(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Ana pencere ayarları
        self.setWindowTitle("FinalModel Tester")
        self.setMinimumSize(1200, 800)  # Resim için daha geniş
        
        # Sistem ve değişkenler
        self.sistem = KonusmaciTanimaSistemi()
        self.model_loaded = False
        self.current_predictions = []
        self.photos_path = "photos"  # Fotoğrafların bulunduğu klasör
        
        # UI ayarla
        self.setup_ui()
        
    def setup_ui(self):
        # Ana widget ve layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Splitter - üst ve alt bölümleri ayırmak için
        splitter = QSplitter(Qt.Vertical)
        
        # Üst panel - kontroller (minimalist)
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(3)
        
        # Kompakt kontrol düzeni
        control_layout = QHBoxLayout()
        
        # Model seçim bölümü
        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout(model_group)
        model_layout.setContentsMargins(5, 10, 5, 5)
        
        self.select_model_btn = QPushButton("Model Seç")
        self.select_model_btn.clicked.connect(self.select_model)
        self.select_model_btn.setFixedWidth(100)
        
        self.model_path_label = QLabel("Seçilmedi")
        self.model_path_label.setStyleSheet("color: gray;")
        
        model_layout.addWidget(self.select_model_btn)
        model_layout.addWidget(self.model_path_label, 1)
        
        # Ses seçim bölümü
        audio_group = QGroupBox("Ses")
        audio_layout = QHBoxLayout(audio_group)
        audio_layout.setContentsMargins(5, 10, 5, 5)
        
        audio_btn_layout = QHBoxLayout()
        
        self.select_file_btn = QPushButton("Dosya")
        self.select_file_btn.clicked.connect(partial(self.select_audio, False))
        self.select_file_btn.setEnabled(False)
        self.select_file_btn.setFixedWidth(80)
        
        self.select_folder_btn = QPushButton("Klasör")
        self.select_folder_btn.clicked.connect(partial(self.select_audio, True))
        self.select_folder_btn.setEnabled(False)
        self.select_folder_btn.setFixedWidth(80)
        
        self.predict_btn = QPushButton("Tahmin")
        self.predict_btn.clicked.connect(self.run_prediction)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setFixedWidth(80)
        self.predict_btn.setStyleSheet("font-weight: bold;")
        
        audio_btn_layout.addWidget(self.select_file_btn)
        audio_btn_layout.addWidget(self.select_folder_btn)
        audio_btn_layout.addWidget(self.predict_btn)
        
        self.audio_path_label = QLabel("Seçilmedi")
        self.audio_path_label.setStyleSheet("color: gray;")
        
        audio_layout.addLayout(audio_btn_layout)
        audio_layout.addWidget(self.audio_path_label, 1)
        
        # İlerleme çubuğu
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(15)
        
        # Üst panele grupları ekle
        control_layout.addWidget(model_group, 1)
        control_layout.addWidget(audio_group, 1)
        
        top_layout.addLayout(control_layout)
        top_layout.addWidget(self.progress_bar)
        
        # Alt panel - sonuçlar (JSON, tablo ve resim geniş)
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        # Alt panel splitter
        bottom_splitter = QSplitter(Qt.Horizontal)
        
        # Sol paneli oluştur - tablo ve resim
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Sonuçlar Tablosu
        results_group = QGroupBox("Tahmin Sonuçları")
        results_layout = QVBoxLayout(results_group)
        results_layout.setContentsMargins(5, 10, 5, 5)
        
        self.results_table = QTableWidget(0, 2)
        self.results_table.setHorizontalHeaderLabels(["Chunk ID", "Kişi"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.itemClicked.connect(self.on_table_item_clicked)
        
        results_layout.addWidget(self.results_table)
        
        # Resim gösterim alanı
        image_group = QGroupBox("Tahmin Edilen Kişi")
        image_layout = QVBoxLayout(image_group)
        image_layout.setContentsMargins(5, 10, 5, 5)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(250, 250)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                background-color: #f5f5f5;
                border-radius: 5px;
            }
        """)
        self.image_label.setText("Kişi resmi burada gösterilecek")
        
        image_layout.addWidget(self.image_label)
        
        # Sol paneli oluştur
        left_layout.addWidget(results_group, 1)
        left_layout.addWidget(image_group, 1)
        
        # JSON Çıktısı - DAHA BÜYÜK
        json_group = QGroupBox("JSON Çıktısı")
        json_layout = QVBoxLayout(json_group)
        json_layout.setContentsMargins(5, 10, 5, 5)
        
        json_btn_layout = QHBoxLayout()
        
        self.copy_json_btn = QPushButton("Kopyala")
        self.copy_json_btn.clicked.connect(self.copy_json)
        self.copy_json_btn.setFixedWidth(80)
        
        self.save_json_btn = QPushButton("Kaydet")
        self.save_json_btn.clicked.connect(self.save_json)
        self.save_json_btn.setFixedWidth(80)
        
        # Boşluk doldurma
        json_btn_layout.addWidget(self.copy_json_btn)
        json_btn_layout.addWidget(self.save_json_btn)
        json_btn_layout.addStretch(1)
        
        self.json_text = QTextEdit()
        self.json_text.setReadOnly(True)
        
        # JSON metnini daha büyük yap
        font = QFont("Courier New", 11)
        self.json_text.setFont(font)
        
        # JSON renklerini iyileştir
        self.json_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                color: #333;
                border: 1px solid #ddd;
            }
        """)
        
        json_layout.addLayout(json_btn_layout)
        json_layout.addWidget(self.json_text)
        
        # Bottom splitter'a panelleri ekle
        bottom_splitter.addWidget(left_panel)
        bottom_splitter.addWidget(json_group)
        
        # Sol panel ve JSON alanı oranını ayarla (1:1)
        bottom_splitter.setSizes([300, 300])
        
        # Alt panele splitter'ı ekle
        bottom_layout.addWidget(bottom_splitter)
        
        # Ana splitter'a panelleri ekle
        splitter.addWidget(top_panel)
        splitter.addWidget(bottom_panel)
        
        # Üst panel minimalist, alt panel daha büyük (1:4 oranı)
        splitter.setSizes([120, 480])
        
        # Ana düzene splitter'ı ekle
        main_layout.addWidget(splitter)
        
        # Ana widget'ı ayarla
        self.setCentralWidget(central_widget)
        
        # Durum çubuğu
        self.statusBar().showMessage("Hazır")
        self.statusBar().setFixedHeight(20)
    
    def on_table_item_clicked(self, item):
        """Tabloda bir öğe tıklandığında resmi göster"""
        row = item.row()
        person_text = self.results_table.item(row, 1).text()
        
        # person1 -> 1 çevirimi
        try:
            person_num = person_text.replace("person", "")
            self.load_person_image(int(person_num))
        except:
            pass
    
    def load_person_image(self, person_num):
        """Kişi resmini yükle ve göster"""
        try:
            image_path = os.path.join(self.photos_path, f"person{person_num}.png")
            
            if os.path.exists(image_path):
                # Pixmap oluştur
                pixmap = QPixmap(image_path)
                
                # Resmi uygun boyuta ölçekle
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                
                # Resmi göster
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText(f"Resim bulunamadı:\n{image_path}")
                
        except Exception as e:
            self.image_label.setText(f"Resim yükleme hatası:\n{str(e)}")
    
    def select_model(self):
        """Modeli yüklemek için dosya seçimi"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Model Dosyası Seç", "", "PyTorch Model (*.pth *.pt)"
        )
        
        if file_path:
            try:
                self.statusBar().showMessage("Model yükleniyor...")
                QApplication.processEvents()
                
                success = self.sistem.modeli_yukle(file_path)
                
                if success:
                    self.model_loaded = True
                    self.model_path_label.setText(os.path.basename(file_path))
                    self.model_path_label.setStyleSheet("color: black;")
                    self.statusBar().showMessage("Model başarıyla yüklendi")
                    
                    # Ses dosyası seçme butonlarını etkinleştir
                    self.select_file_btn.setEnabled(True)
                    self.select_folder_btn.setEnabled(True)
                else:
                    QMessageBox.critical(self, "Hata", "Model yüklenemedi!")
                    self.statusBar().showMessage("Model yükleme hatası")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Model yükleme hatası: {str(e)}")
                self.statusBar().showMessage("Model yükleme hatası")
    
    def select_audio(self, is_folder=False):
        """Ses dosyası veya klasör seç"""
        if is_folder:
            folder_path = QFileDialog.getExistingDirectory(
                self, "Ses Dosyaları Klasörünü Seç"
            )
            if folder_path:
                self.audio_path_label.setText(os.path.basename(folder_path))
                self.audio_path_label.setStyleSheet("color: black;")
                self.audio_path = folder_path
                self.is_folder = True
                self.predict_btn.setEnabled(True)
                self.statusBar().showMessage(f"Klasör: {os.path.basename(folder_path)}")
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Ses Dosyasını Seç", "", 
                "Ses Dosyaları (*.wav *.mp3 *.m4a *.flac *.ogg *.wma);;Tüm Dosyalar (*.*)"
            )
            if file_path:
                self.audio_path_label.setText(os.path.basename(file_path))
                self.audio_path_label.setStyleSheet("color: black;")
                self.audio_path = file_path
                self.is_folder = False
                self.predict_btn.setEnabled(True)
                self.statusBar().showMessage(f"Dosya: {os.path.basename(file_path)}")
    
    def run_prediction(self):
        """Tahmin işlemini başlat"""
        if not self.model_loaded or not hasattr(self, 'audio_path'):
            QMessageBox.warning(self, "Uyarı", "Önce model ve ses dosyası/klasörü seçin!")
            return
        
        # UI elemanlarını devre dışı bırak
        self.predict_btn.setEnabled(False)
        self.select_file_btn.setEnabled(False)
        self.select_folder_btn.setEnabled(False)
        self.select_model_btn.setEnabled(False)
        
        # Progress bar'ı sıfırla
        self.progress_bar.setValue(0)
        
        # Durum çubuğunu güncelle
        self.statusBar().showMessage("Tahmin yapılıyor...")
        
        # Tahmin thread'ini başlat
        self.tahmin_thread = TahminThread(self.sistem, self.audio_path, self.is_folder)
        self.tahmin_thread.progress.connect(self.update_progress)
        self.tahmin_thread.finished.connect(self.on_prediction_finished)
        self.tahmin_thread.error.connect(self.on_prediction_error)
        self.tahmin_thread.start()
    
    @Slot(int)
    def update_progress(self, value):
        """İlerleme çubuğunu güncelle"""
        self.progress_bar.setValue(value)
    
    @Slot(list)
    def on_prediction_finished(self, predictions):
        """Tahmin tamamlandığında"""
        # Sonuçları kaydet
        self.current_predictions = predictions
        
        # Sonuçlar tablosunu temizle ve yeniden doldur
        self.results_table.setRowCount(0)
        for i, pred in enumerate(predictions):
            self.results_table.insertRow(i)
            self.results_table.setItem(i, 0, QTableWidgetItem(pred["id"]))
            self.results_table.setItem(i, 1, QTableWidgetItem(pred["tahmin_etiketi"]))
        
        # JSON oluştur ve göster
        self.update_json_output()
        
        # İlk tahmin varsa resmini göster
        if predictions:
            try:
                person_num = int(predictions[0]["tahmin_etiketi"].replace("person", ""))
                self.load_person_image(person_num)
            except:
                pass
        
        # UI elemanlarını tekrar etkinleştir
        self.predict_btn.setEnabled(True)
        self.select_file_btn.setEnabled(True)
        self.select_folder_btn.setEnabled(True)
        self.select_model_btn.setEnabled(True)
        
        # Durum çubuğunu güncelle
        self.statusBar().showMessage(f"Tahmin tamamlandı: {len(predictions)} sonuç")
    
    @Slot(str)
    def on_prediction_error(self, error_message):
        """Tahmin sırasında hata oluştuğunda"""
        QMessageBox.critical(self, "Hata", error_message)
        
        # UI elemanlarını tekrar etkinleştir
        self.predict_btn.setEnabled(True)
        self.select_file_btn.setEnabled(True)
        self.select_folder_btn.setEnabled(True)
        self.select_model_btn.setEnabled(True)
        
        # Durum çubuğunu güncelle
        self.statusBar().showMessage("Tahmin hatası")
    
    def update_json_output(self):
        """JSON çıktısını güncelle"""
        json_data = {
            "yarisma": YARISMA_ADI,
            "takim_adi": TAKIM_ADI,
            "takim_id": TAKIM_ID,
            "tahminler": self.current_predictions
        }
        
        # JSON'u formatlı şekilde göster
        json_string = json.dumps(json_data, indent=2)
        self.json_text.setText(json_string)
    
    def copy_json(self):
        """JSON çıktısını panoya kopyala"""
        if self.json_text.toPlainText():
            clipboard = QApplication.clipboard()
            clipboard.setText(self.json_text.toPlainText())
            self.statusBar().showMessage("JSON kopyalandı", 3000)
    
    def save_json(self):
        """JSON çıktısını dosyaya kaydet"""
        if not self.json_text.toPlainText():
            QMessageBox.warning(self, "Uyarı", "Kaydedilecek tahmin sonucu yok!")
            return
        
        # Kaydetme diyaloğu
        file_path, _ = QFileDialog.getSaveFileName(
            self, "JSON Dosyasını Kaydet", 
            f"tahminler_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json", 
            "JSON Dosyaları (*.json)"
        )
        
        if file_path:
            try:
                # JSON verisi oluştur
                json_data = {
                    "yarisma": YARISMA_ADI,
                    "takim_adi": TAKIM_ADI,
                    "takim_id": TAKIM_ID,
                    "tahminler": self.current_predictions
                }
                
                # Dosyaya kaydet
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2)
                
                self.statusBar().showMessage(f"JSON kaydedildi: {os.path.basename(file_path)}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Dosya kaydedilemedi: {str(e)}")

# Uygulama başlatma
if __name__ == "__main__":
    # Uygulamayı başlat
    app = QApplication(sys.argv)
    
    # Fusion stilini kullan - daha modern görünüm
    app.setStyle("Fusion")
    
    # Ana pencereyi göster
    window = FinalModelTester()
    window.show()
    
    sys.exit(app.exec())