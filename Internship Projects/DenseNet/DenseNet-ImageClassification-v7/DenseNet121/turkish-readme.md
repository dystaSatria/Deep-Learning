# DenseNet121 Hibrit Sınıflandırma Sistemi

Bu proje, görüntü sınıflandırma görevleri için DenseNet121 özellik çıkarıcısı ve çeşitli makine öğrenimi sınıflandırıcılarını birleştiren kapsamlı bir hibrit sistemdir. Bu sistem, transfer öğrenimi tekniklerini kullanarak karmaşık görüntü sınıflandırma problemlerini çözmek için tasarlanmıştır.

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Özellikler](#özellikler)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Sistem Mimarisi](#sistem-mimarisi)
- [Modeller](#modeller)
- [Değerlendirme Metrikleri](#değerlendirme-metrikleri)
- [Görselleştirme](#görselleştirme)
- [Lisans](#lisans)

## Genel Bakış

Bu proje, görüntü sınıflandırma için DenseNet121 CNN (Evrişimli Sinir Ağı) modelini özellik çıkarıcı olarak kullanan ve daha sonra SVM, Gradient Boosting, XGBoost, LightGBM, CatBoost ve AdaBoost gibi çeşitli makine öğrenimi algoritmaları ile bu özellikleri işleyen hibrit bir yaklaşım sunmaktadır. Sistem, transfer öğrenimi ilkelerini uygulayarak, bilinen güçlü bir derin öğrenme mimarisinin (DenseNet121) özellik çıkarma yeteneklerini çeşitli geleneksel makine öğrenimi algoritmaları ile birleştirerek en iyi performansı elde etmeyi amaçlar.

## Özellikler

- **Transfer Öğrenimi**: ImageNet üzerinde önceden eğitilmiş DenseNet121 modelini kullanır
- **Çoklu Sınıflandırıcı Desteği**: SVM (Linear, RBF, Poly, Sigmoid), Gradient Boosting, AdaBoost, XGBoost, LightGBM ve CatBoost
- **Kapsamlı Değerlendirme**: Doğruluk, kesinlik, duyarlılık, F1-skoru, F2-skoru, F0-skoru, özgüllük, MCC ve Kappa dahil olmak üzere geniş bir metrik yelpazesini değerlendirir
- **Görselleştirme Araçları**: Eğitim geçmişi, karışıklık matrisleri, ROC eğrileri ve hassasiyet-geri çağırma eğrileri için detaylı görselleştirmeler
- **Sınıf Bazlı Metrikler**: Her sınıf için detaylı performans metrikleri
- **Kolay Kullanım**: Farklı veri kümeleri için kullanımı kolay olan yüksek düzeyde bir API

## Gereksinimler

Sistemin çalışması için aşağıdaki kütüphanelere ihtiyaç vardır:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow (>= 2.0)
```

İsteğe bağlı olarak şu gelişmiş sınıflandırıcılar da kurulabilir:
```
xgboost
lightgbm
catboost
```

## Kurulum

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

2. İsteğe bağlı boosting kütüphanelerini yükleyin:

```bash
pip install xgboost lightgbm catboost
```

## Kullanım

Temel kullanım örneği:

```python
from densenet_hybrid_classification import run_enhanced_classification

# Veri klasörü yolunu belirtin
DATA_FOLDER = "/verinizin/yolu"

# Gelişmiş sınıflandırmayı çalıştırın
results = run_enhanced_classification(DATA_FOLDER)

# Sonuçları gösterin
print(f"En iyi hibrit model: {results['best_hybrid_model_name']} - F1 skoru: {results['best_hybrid_model_f1']:.4f}")
print(f"DenseNet121 (Standart) F1 skoru: {results['densenet_f1']:.4f}")
```

## Sistem Mimarisi

Sistem şu temel bileşenlerden oluşur:

1. **Veri İşleme ve Artırma**: Görüntüleri yüklemek, ölçeklendirmek ve artırmak için `ImageDataGenerator` kullanır
2. **Özellik Çıkarma**: Görüntü özelliklerini çıkarmak için önceden eğitilmiş DenseNet121 kullanır
3. **Sınıflandırıcı Eğitimi**: Çıkarılan özellikler üzerinde çeşitli makine öğrenimi sınıflandırıcıları eğitir
4. **Model Değerlendirmesi**: Tüm modelleri kapsamlı bir metrik kümesi kullanarak değerlendirir
5. **Görselleştirme**: Eğitim sürecini ve sonuçları analiz etmek için çeşitli grafikler oluşturur

## Modeller

Bu proje aşağıdaki modelleri uygular ve değerlendirir:

1. **DenseNet121 (Standart)**: Temel derin öğrenme modeli
2. **SVM Modelleri**:
   - DenseNet121 + SVM (Linear)
   - DenseNet121 + SVM (RBF)
   - DenseNet121 + SVM (Poly)
   - DenseNet121 + SVM (Sigmoid)
3. **Boosting Modelleri**:
   - DenseNet121 + Gradient Boosting
   - DenseNet121 + XGBoost (mevcutsa)
   - DenseNet121 + LightGBM (mevcutsa)
   - DenseNet121 + CatBoost (mevcutsa)
   - DenseNet121 + AdaBoost

## Değerlendirme Metrikleri

Değerlendirme için kullanılan metrikler:

- **Doğruluk (Accuracy)**: Doğru tahminlerin toplam tahminlere oranı
- **Kesinlik (Precision)**: Doğru pozitif tahminlerin toplam pozitif tahminlere oranı
- **Duyarlılık/Geri Çağırma (Recall/Sensitivity)**: Doğru pozitif tahminlerin gerçek pozitiflere oranı
- **Özgüllük (Specificity)**: Doğru negatif tahminlerin gerçek negatiflere oranı
- **F-Skorları**: Kesinlik ve duyarlılığın harmonik ortalamaları (F1, F2, F0)
- **Matthews Korelasyon Katsayısı (MCC)**: Sınıflandırma kalitesinin ölçümü
- **Cohen's Kappa**: Rastgele şansa göre düzeltilmiş doğruluk

## Görselleştirme

Kod, analiz için aşağıdaki görselleştirmeleri oluşturur:

1. **Eğitim Geçmişi**: Doğruluk ve kayıp eğrileri
2. **Karışıklık Matrisi**: Her model için sınıflandırma performansının detaylı görünümü
3. **ROC Eğrileri**: Alıcı İşletim Karakteristik eğrileri ve AUC değerleri
4. **Hassasiyet-Geri Çağırma Eğrileri**: En iyi model için hassasiyet-geri çağırma eğrisi
5. **Metrik Karşılaştırma Grafikleri**: Tüm modeller ve metrikler arasında görsel karşılaştırma

## Çıktılar

Proje şu dosyaları oluşturur:

- **densenet121_training_history.png**: Eğitim doğruluğu ve kayıp eğrileri
- **densenet121_confusion_matrix.png**: DenseNet121 için karışıklık matrisi
- **best_hybrid_confusion_matrix.png**: En iyi hibrit model için karışıklık matrisi
- **roc_curves_comparison.png**: Tüm modeller için ROC eğrileri
- **roc_auc_comparison.csv**: ROC AUC değerleri karşılaştırması
- **precision_recall_curve_best_model.png**: En iyi model için hassasiyet-geri çağırma eğrisi
- **metrics_comparison.png**: Tüm modeller için metrik karşılaştırması
- **models_comparison.png**: Tüm metrikler için model karşılaştırması
- **densenet_per_class_metrics.csv**: DenseNet121 için sınıf bazlı metrikler
- **best_hybrid_per_class_metrics.csv**: En iyi hibrit model için sınıf bazlı metrikler
- **per_class_metrics_tables.png**: Sınıf bazlı metrik tabloları
- **all_models_metrics.csv**: Tüm model metriklerinin özeti

## Lisans

Bu proje [buraya lisansınızı ekleyin, örn. MIT] lisansı altında lisanslanmıştır.