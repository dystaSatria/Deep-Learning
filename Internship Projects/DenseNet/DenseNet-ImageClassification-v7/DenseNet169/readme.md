# DenseNet169 Hibrit Sınıflandırma Kodu Detaylı Açıklaması


Bu belge, DenseNet169 ve çeşitli makine öğrenimi sınıflandırıcılarını birleştiren görüntü sınıflandırma sisteminin kodunu ayrıntılı olarak açıklamaktadır.

## İçindekiler

1. [Kütüphaneler ve Bağımlılıklar](#kütüphaneler-ve-bağımlılıklar)
2. [Sabitler ve Konfigürasyon](#sabitler-ve-konfigürasyon)
3. [Metrik Hesaplama Fonksiyonları](#metrik-hesaplama-fonksiyonları)
4. [Özellik Çıkarma Fonksiyonları](#özellik-çıkarma-fonksiyonları)
5. [DenseNet169 Model Oluşturma](#densenet169-model-oluşturma)
6. [Eğitim ve Değerlendirme Fonksiyonları](#eğitim-ve-değerlendirme-fonksiyonları)
7. [Sınıflandırıcı Fonksiyonları](#sınıflandırıcı-fonksiyonları)
8. [Görselleştirme Fonksiyonları](#görselleştirme-fonksiyonları)
9. [Ana Çalıştırma Fonksiyonu](#ana-çalıştırma-fonksiyonu)
10. [Kullanım Örneği](#kullanım-örneği)

## Kütüphaneler ve Bağımlılıklar

Kod, aşağıdaki kütüphaneleri içe aktarır:

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, matthews_corrcoef,
    cohen_kappa_score, fbeta_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
import tensorflow as tf
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
```

Kod ayrıca aşağıdaki opsiyonel kütüphaneleri içe aktarmaya çalışır:

```python
try:
    import lightgbm as lgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM is not installed. Skipping LightGBM classifier.")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except (ImportError, ValueError) as e:
    CATBOOST_AVAILABLE = False
    print(f"Warning: CatBoost cannot be used. Error: {e}. Skipping CatBoost classifier.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost is not installed. Skipping XGBoost classifier.")
```

Bu yapı, belirli kütüphaneler yüklü değilse bile kodun çalışmaya devam etmesini sağlar; sadece ilgili sınıflandırıcılar atlanır.

Rastgeleliği kontrol etmek için rastgele tohum değerleri ayarlanır:

```python
import random as python_random
np.random.seed(42)
python_random.seed(42)
tf.random.set_seed(42)
```

## Sabitler ve Konfigürasyon

Kod, aşağıdaki sabit değerleri tanımlar:

```python
IMAGE_SIZE = (150, 150)  # Girdi görüntülerinin boyutu
BATCH_SIZE = 32          # Eğitim batch boyutu
EPOCHS = 30              # Maksimum epoch sayısı
LEARNING_RATE = 0.0001   # Öğrenme oranı
VAL_SPLIT = 0.2          # Doğrulama veri kümesi için ayrılan oran
```

## Metrik Hesaplama Fonksiyonları

### 1. calculate_specificity

Bu fonksiyon, çok sınıflı sınıflandırma için özgüllük (specificty) metriğini hesaplar.

```python
def calculate_specificity(y_true, y_pred):
    """Calculate specificity for multi-class classification

    Specificity = TN / (TN + FP)
    """
    cm = confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (fp + np.diag(cm) + cm.sum(axis=1) - np.diag(cm))
    specificity = np.zeros_like(tn, dtype=float)
    for i in range(len(specificity)):
        if tn[i] + fp[i] > 0:
            specificity[i] = tn[i] / (tn[i] + fp[i])
        else:
            specificity[i] = 0.0

    # Ağırlıklı ortalama özgüllük değerini döndür
    return np.average(specificity, weights=np.bincount(y_true) if len(np.unique(y_true)) > 1 else None)
```

### 2. calculate_metrics

Bu fonksiyon, tüm gerekli metrikleri hesaplar:

```python
def calculate_metrics(y_true, y_pred):
    """Calculate all required metrics"""
    metrics = {}

    # Temel metrikler
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')

    # F-skorları
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['f2'] = fbeta_score(y_true, y_pred, beta=2, average='weighted')
    metrics['f0'] = fbeta_score(y_true, y_pred, beta=0.5, average='weighted')

    # Özgüllük - kendi özel fonksiyonumuzu kullanarak
    metrics['specificity'] = calculate_specificity(y_true, y_pred)

    # İleri düzey metrikler
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)

    return metrics
```

### 3. calculate_per_class_metrics

Bu fonksiyon, her sınıf için metrikleri hesaplar:

```python
def calculate_per_class_metrics(y_true, y_pred, class_names):
    """Calculate metrics for each class"""
    n_classes = len(class_names)

    # Sonuçlar sözlüğünü başlat
    results = {
        'class': class_names,
        'accuracy': [],
        'precision': [],
        'recall': [],
        'specificity': [],
        'f0': [],
        'f1': [],
        'f2': [],
        'kappa': [],
        'mcc': []
    }

    # Sınıf bazlı hesaplamalar için one-hot kodlamaya dönüştür
    y_true_bin = np.zeros((len(y_true), n_classes))
    y_pred_bin = np.zeros((len(y_pred), n_classes))

    for i in range(len(y_true)):
        y_true_bin[i, y_true[i]] = 1

    for i in range(len(y_pred)):
        y_pred_bin[i, y_pred[i]] = 1

    # Karışıklık matrisini hesapla
    cm = confusion_matrix(y_true, y_pred)

    # Her sınıf için metrikleri hesapla
    for i in range(n_classes):
        # İkili metrikler için, mevcut sınıfı pozitif ve diğer tüm sınıfları negatif olarak ele alırız
        tp = cm[i, i]  # Doğru pozitifler
        fp = np.sum(cm[:, i]) - tp  # Yanlış pozitifler
        fn = np.sum(cm[i, :]) - tp  # Yanlış negatifler
        tn = np.sum(cm) - tp - fp - fn  # Doğru negatifler

        # Doğruluk
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        results['accuracy'].append(accuracy)

        # Kesinlik
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        results['precision'].append(precision)

        # Duyarlılık / Hassasiyet
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        results['recall'].append(recall)

        # Özgüllük
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        results['specificity'].append(specificity)

        # F-ölçümleri
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            f2 = 5 * precision * recall / (4 * precision + recall)
            f0 = 1.25 * precision * recall / (0.25 * precision + recall)
        else:
            f1 = f2 = f0 = 0

        results['f1'].append(f1)
        results['f2'].append(f2)
        results['f0'].append(f0)

        # MCC ve Kappa için, sklearn fonksiyonlarını ikili sınıflandırma üzerinde kullanacağız
        # mevcut sınıfı pozitif ve diğer tüm sınıfları negatif olarak ele alarak
        y_true_class = (y_true == i).astype(int)
        y_pred_class = (y_pred == i).astype(int)

        results['mcc'].append(matthews_corrcoef(y_true_class, y_pred_class))
        results['kappa'].append(cohen_kappa_score(y_true_class, y_pred_class))

    return pd.DataFrame(results)
```

## Özellik Çıkarma Fonksiyonları

### 1. create_feature_extractor

Bu fonksiyon, DenseNet169'den özellik çıkarmak için bir model oluşturur:

```python
def create_feature_extractor():
    # Önceden eğitilmiş ağırlıklarla DenseNet169'i yükle
    base_model = DenseNet169(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    # Tüm temel model katmanlarını dondur
    for layer in base_model.layers:
        layer.trainable = False

    # Global havuzlama katmanı ekle
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Model oluştur
    model = Model(inputs=base_model.input, outputs=x)

    return model
```

### 2. extract_features

Bu fonksiyon, veri üreteçlerinden özellikler çıkarır:

```python
def extract_features(feature_extractor, data_generator):
    features = []
    labels = []

    # Tüm batch'ler için özellikler çıkar
    num_batches = len(data_generator)
    for i in range(num_batches):
        x_batch, y_batch = data_generator[i]
        batch_features = feature_extractor.predict(x_batch, verbose=0)
        features.append(batch_features)
        labels.append(np.argmax(y_batch, axis=1))

    # Özellikleri ve etiketleri birleştir
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels
```

## DenseNet169 Model Oluşturma

### build_densenet_model

Bu fonksiyon, ReLU aktivasyonu ve 1024 nöronlu Yoğun katman içeren modifiye edilmiş bir DenseNet169 modeli oluşturur:

```python
def build_densenet_model(num_classes):
    # Önceden eğitilmiş ağırlıklarla DenseNet169'i yükle
    base_model = DenseNet169(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    # Tüm temel model katmanlarını dondur
    for layer in base_model.layers:
        layer.trainable = False

    # Sınıflandırma katmanları ekle
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Yoğun katmanı 1024 nöron ile açıkça ReLU aktivasyonunu kullanarak değiştir
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # RMSprop optimizer ve belirtilen öğrenme oranıyla derle
    model.compile(
        optimizer=RMSprop(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
```

## Eğitim ve Değerlendirme Fonksiyonları

### 1. fast_train

Bu fonksiyon, geçmiş döndüren basitleştirilmiş bir eğitim fonksiyonudur:

```python
def fast_train(model, train_generator, validation_generator):
    # Aşırı uyumu önlemek için erken durdurma
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Modifiye edilmiş epoch'larla eğitim
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1
    )

    return model, history
```

### 2. evaluate_densenet

Bu fonksiyon, temel DenseNet169 için değerlendirme yapar ve tahminler getirir:

```python
def evaluate_densenet(model, test_generator):
    # Tahminleri al
    y_pred_prob = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_generator.classes

    # Metrikleri hesapla
    metrics = calculate_metrics(y_true, y_pred)

    return metrics, y_true, y_pred, y_pred_prob
```

## Sınıflandırıcı Fonksiyonları

### 1. SVM Sınıflandırıcı

```python
def train_and_evaluate_svm(train_features, train_labels, test_features, test_labels, kernel='linear'):
    print(f"Training SVM with {kernel} kernel...")

    # SVM sınıflandırıcı oluştur ve eğit
    svm = SVC(kernel=kernel, probability=True, random_state=42)
    svm.fit(train_features, train_labels)

    # Tahminleri al
    y_pred = svm.predict(test_features)
    y_pred_prob = svm.predict_proba(test_features)

    # Metrikleri hesapla
    metrics = calculate_metrics(test_labels, y_pred)

    # Metrikler sözlüğüne model adını ekle
    metrics['model'] = f"DenseNet169 + SVM ({kernel})"

    return metrics, y_pred, y_pred_prob
```

### 2. Gradient Boosting Sınıflandırıcı

```python
def train_and_evaluate_gradient_boosting(train_features, train_labels, test_features, test_labels):
    print("Training Gradient Boosting classifier...")

    # Gradient Boosting sınıflandırıcı oluştur ve eğit
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(train_features, train_labels)

    # Tahminleri al
    y_pred = gb.predict(test_features)
    y_pred_prob = gb.predict_proba(test_features)

    # Metrikleri hesapla
    metrics = calculate_metrics(test_labels, y_pred)

    # Metrikler sözlüğüne model adını ekle
    metrics['model'] = "DenseNet169 + Gradient Boosting"

    return metrics, y_pred, y_pred_prob
```

### 3. LightGBM Sınıflandırıcı

```python
def train_and_evaluate_lightgbm(train_features, train_labels, test_features, test_labels):
    print("Training LightGBM classifier...")

    # LightGBM sınıflandırıcı oluştur ve eğit
    lgb = lgbm.LGBMClassifier(n_estimators=100, random_state=42)
    lgb.fit(train_features, train_labels)

    # Tahminleri al
    y_pred = lgb.predict(test_features)
    y_pred_prob = lgb.predict_proba(test_features)

    # Metrikleri hesapla
    metrics = calculate_metrics(test_labels, y_pred)

    # Metrikler sözlüğüne model adını ekle
    metrics['model'] = "DenseNet169 + LightGBM"

    return metrics, y_pred, y_pred_prob
```

### 4. XGBoost Sınıflandırıcı

```python
def train_and_evaluate_xgboost(train_features, train_labels, test_features, test_labels):
    print("Training XGBoost classifier...")

    # XGBoost sınıflandırıcı oluştur ve eğit
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=len(np.unique(train_labels)),
        random_state=42,
        eval_metric='mlogloss'
    )

    xgb_classifier.fit(train_features, train_labels)

    # Tahminleri al
    y_pred = xgb_classifier.predict(test_features)
    y_pred_prob = xgb_classifier.predict_proba(test_features)

    # Metrikleri hesapla
    metrics = calculate_metrics(test_labels, y_pred)

    # Metrikler sözlüğüne model adını ekle
    metrics['model'] = "DenseNet169 + XGBoost"

    return metrics, y_pred, y_pred_prob
```

### 5. CatBoost Sınıflandırıcı

```python
def train_and_evaluate_catboost(train_features, train_labels, test_features, test_labels):
    print("Training CatBoost classifier...")

    # Çalışma zamanında CatBoost'un gerçekten mevcut olup olmadığını kontrol et
    if 'CatBoostClassifier' not in globals():
        print("Error: CatBoostClassifier is not available. Cannot proceed with CatBoost training.")
        # Sıfırlarla kukla metrikler döndür
        metrics = {
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0,
            'f2': 0, 'f0': 0, 'specificity': 0, 'mcc': 0, 'kappa': 0,
            'model': "DenseNet169 + CatBoost (FAILED)"
        }
        return metrics, None, None

    try:
        # CatBoost sınıflandırıcıyı verbose=0 ile çıktıyı azaltmak için oluştur ve eğit
        # ve daha hızlı eğitim için iterations=100
        cb = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            loss_function='MultiClass',
            random_seed=42,
            verbose=0
        )
        cb.fit(train_features, train_labels)

        # Tahminleri al
        y_pred = cb.predict(test_features)
        y_pred_prob = cb.predict_proba(test_features)

        # Metrikleri hesapla
        metrics = calculate_metrics(test_labels, y_pred)

        # Metrikler sözlüğüne model adını ekle
        metrics['model'] = "DenseNet169 + CatBoost"

        return metrics, y_pred, y_pred_prob

    except Exception as e:
        print(f"Error during CatBoost training/evaluation: {e}")
        # Sıfırlarla kukla metrikler döndür
        metrics = {
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0,
            'f2': 0, 'f0': 0, 'specificity': 0, 'mcc': 0, 'kappa': 0,
            'model': "DenseNet169 + CatBoost "
        }
        return metrics, None, None
```

### 6. AdaBoost Sınıflandırıcı

```python
def train_and_evaluate_adaboost(train_features, train_labels, test_features, test_labels):
    print("Training AdaBoost classifier...")

    # AdaBoost sınıflandırıcı oluştur ve eğit
    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    ada.fit(train_features, train_labels)

    # Tahminleri al
    y_pred = ada.predict(test_features)
    y_pred_prob = ada.predict_proba(test_features)

    # Metrikleri hesapla
    metrics = calculate_metrics(test_labels, y_pred)

    # Metrikler sözlüğüne model adını ekle
    metrics['model'] = "DenseNet169 + AdaBoost"

    return metrics, y_pred, y_pred_prob
```

## Görselleştirme Fonksiyonları

### 1. plot_training_history

Bu fonksiyon, eğitim geçmişini görselleştirir:

```python
def plot_training_history(history):
    """Plot training and validation accuracy and loss curves"""
    # 2 alt grafikli bir figür oluştur
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Eğitim ve doğrulama doğruluğunu çizdir
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('DenseNet169 Model Accuracy', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.legend(['Train', 'Validation'], loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Eğitim ve doğrulama kaybını çizdir
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('DenseNet169 Model Loss', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(['Train', 'Validation'], loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('densenet169_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig
```

### 2. plot_confusion_matrix

Bu fonksiyon, karışıklık matrisini çizdirir:

```python
def plot_confusion_matrix(y_true, y_pred, class_names, title):
    """Plot confusion matrix with custom styling"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    # Figürü kaydet
    filename = title.replace(' ', '_').lower() + '.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    return cm
```

### 3. calculate_and_plot_roc

Bu fonksiyon, ROC eğrilerini hesaplar ve çizdirir:

```python
def calculate_and_plot_roc(all_models_results, class_names, num_classes):
    plt.figure(figsize=(15, 12))

    # Tüm modeller için ROC AUC değerlerini sakla
    roc_aucs = {}

    # Her model için
    for model_name, model_data in all_models_results.items():
        y_true = model_data['y_true']
        y_prob = model_data['y_prob']

        # Her sınıf için ROC eğrisi ve ROC alanını hesapla
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # ROC hesaplaması için etiketleri one-hot kodla
        y_true_bin = np.zeros((len(y_true), num_classes))
        for i in range(len(y_true)):
            y_true_bin[i, y_true[i]] = 1

        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Mikro-ortalama ROC eğrisi ve ROC alanını hesapla
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Mikro-ortalama ROC AUC'yi sakla
        roc_aucs[model_name] = roc_auc["micro"]

        # ROC eğrisini çizdir
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'{model_name} (AUC = {roc_auc["micro"]:.4f})',
                 linewidth=2)

    # Diyagonal çizgiyi çizdir
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves for All Models', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ROC AUC karşılaştırma tablosu oluştur
    roc_df = pd.DataFrame({
        'Model': list(roc_aucs.keys()),
        'ROC AUC': [f"{v:.4f}" for v in roc_aucs.values()]
    })

    # ROC AUC'ye göre sırala
    roc_df = roc_df.sort_values('ROC AUC', ascending=False).reset_index(drop=True)

    # CSV'ye kaydet
    roc_df.to_csv('roc_auc_comparison.csv', index=False)

    # ROC AUC karşılaştırmasını göster
    print("\n===== ROC AUC Comparison =====")
    print(roc_df)

    return roc_df, roc_aucs
```

### 4. plot_precision_recall_curve

Bu fonksiyon, en iyi model için hassasiyet-geri çağırma eğrisini çizdirir:

```python
def plot_precision_recall_curve(y_true, y_pred_prob, class_names, num_classes, model_name):
    """Plot precision-recall curve for the best model"""
    plt.figure(figsize=(15, 12))

    # Etiketleri one-hot kodla
    y_true_bin = np.zeros((len(y_true), num_classes))
    for i in range(len(y_true)):
        y_true_bin[i, y_true[i]] = 1

    # Mikro-ortalama hassasiyet-geri çağırma eğrisini hesapla
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_prob[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_prob[:, i])

    # Mikro-ortalama hassasiyet-geri çağırma eğrisini hesapla
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_pred_prob.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_true_bin.ravel(), y_pred_prob.ravel()
    )

    # Mikro-ortalama hassasiyet-geri çağırma eğrisini çizdir
    plt.plot(recall["micro"], precision["micro"],
             label=f'Micro-average (AP = {average_precision["micro"]:.2f})',
             linewidth=2)

    # Her sınıf için hassasiyet-geri çağırma eğrisini çizdir
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for i, color in zip(range(num_classes), colors):
        plt.plot(recall[i], precision[i], color=color, linewidth=2,
                 label=f'{class_names[i]} (AP = {average_precision[i]:.2f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.savefig('precision_recall_curve_best_model.png', dpi=300, bbox_inches='tight')
    plt.show()

    return average_precision
```

### 5. create_metrics_tables

Bu fonksiyon, sınıf başına metrik tablolarını oluşturur ve kaydeder:

```python
def create_metrics_tables(densenet_per_class_df, best_hybrid_per_class_df):
    """Create and save tables with per-class metrics"""
    # DenseNet tablosunu biçimlendir
    densenet_table = densenet_per_class_df.copy()
    for col in ['accuracy', 'precision', 'recall', 'specificity', 'f0', 'f1', 'f2', 'kappa', 'mcc']:
        densenet_table[col] = densenet_table[col].map(lambda x: f"{x:.4f}")

    # En İyi Hibrit tablosunu biçimlendir
    best_hybrid_table = best_hybrid_per_class_df.copy()
    for col in ['accuracy', 'precision', 'recall', 'specificity', 'f0', 'f1', 'f2', 'kappa', 'mcc']:
        best_hybrid_table[col] = best_hybrid_table[col].map(lambda x: f"{x:.4f}")

    # Tabloları CSV'ye kaydet
    densenet_table.to_csv('densenet_per_class_metrics.csv', index=False)
    best_hybrid_table.to_csv('best_hybrid_per_class_metrics.csv', index=False)

    # Görüntülemek için görsel tablolar oluştur
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # DenseNet tablosu
    ax1.axis('tight')
    ax1.axis('off')
    ax1_table = ax1.table(
        cellText=densenet_table[['class', 'accuracy', 'precision', 'recall', 'specificity',
                             'f0', 'f1', 'f2', 'kappa', 'mcc']].values,
        colLabels=['Class Names', 'A', 'P', 'R', 'S', 'F0', 'F1', 'F2', 'Kappa', 'MCC'],
        cellLoc='center',
        loc='center'
    )
    ax1_table.auto_set_font_size(False)
    ax1_table.set_fontsize(9)
    ax1_table.scale(1, 1.5)
    ax1.set_title('DenseNet169 Per-Class Metrics', fontsize=14)

    # En İyi Hibrit tablosu
    ax2.axis('tight')
    ax2.axis('off')
    ax2_table = ax2.table(
        cellText=best_hybrid_table[['class', 'accuracy', 'precision', 'recall', 'specificity',
                                'f0', 'f1', 'f2', 'kappa', 'mcc']].values,
        colLabels=['Class Names', 'A', 'P', 'R', 'S', 'F0', 'F1', 'F2', 'Kappa', 'MCC'],
        cellLoc='center',
        loc='center'
    )
    ax2_table.auto_set_font_size(False)
    ax2_table.set_fontsize(9)
    ax2_table.scale(1, 1.5)
    ax2.set_title('DenseNet169 + Best Hybrid Model Per-Class Metrics', fontsize=14)

    plt.tight_layout()
    plt.savefig('per_class_metrics_tables.png', dpi=300, bbox_inches='tight')
    plt.show()

    return densenet_table, best_hybrid_table
```

### 6. visualize_metrics

Bu fonksiyon, karşılaştırmalı bir grafikte tüm metrikleri görselleştirir:

```python
def visualize_metrics(all_metrics_df):
    # Daha kolay çizdirmek için veri çerçevesini erit
    melted_df = pd.melt(all_metrics_df, id_vars=['model'],
                         value_vars=['accuracy', 'precision', 'recall', 'f1', 'f2', 'f0',
                                    'specificity', 'mcc', 'kappa'],
                         var_name='metric', value_name='value')

    # Modele göre metrikleri çizdir
    plt.figure(figsize=(20, 12))
    sns.barplot(x='model', y='value', hue='metric', data=melted_df)
    plt.title('Comparison of Metrics Across All Models', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric', title_fontsize=12, fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Metrik türüne göre metrikleri çizdir
    plt.figure(figsize=(20, 12))
    sns.barplot(x='metric', y='value', hue='model', data=melted_df)
    plt.title('Comparison of Models Across All Metrics', fontsize=16)
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(title='Model', title_fontsize=12, fontsize=10, loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.tight_layout()
    plt.savefig('models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
```

## Ana Çalıştırma Fonksiyonu

### run_enhanced_classification

Bu fonksiyon, sabit en iyi hibrit model seçimi ile geliştirilmiş bir yürütme fonksiyonudur:

```python
def run_enhanced_classification(data_folder):
    print(f"Starting enhanced DenseNet169 classification with multiple classifiers on {data_folder}...")

    # Veri klasörünün var olup olmadığını kontrol et
    if not os.path.exists(data_folder):
        print(f"Error: Data folder '{data_folder}' not found.")
        return None

    # Daha basit artırma ile veri üreteçleri oluştur
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=VAL_SPLIT
    )

    # Klasör yapısını kontrol et
    if os.path.exists(os.path.join(data_folder, 'train')):
        # Train/test yapısına sahip
        train_folder = os.path.join(data_folder, 'train')
        test_folder = os.path.join(data_folder, 'test') if os.path.exists(os.path.join(data_folder, 'test')) else None
    else:
        # Sınıf alt klasörlerine sahip tek klasör
        train_folder = data_folder
        test_folder = None

    # Üreteçler oluştur
    print("Creating data generators...")
    train_generator = datagen.flow_from_directory(
        train_folder,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        train_folder,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        subset='validation'
    )

    # Test verileri için
    if test_folder:
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_folder,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
    else:
        # Test verileri olarak doğrulama verilerini kullan
        test_generator = validation_generator

    # Sınıf adlarını ve sınıf sayısını al
    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # Tüm modeller için sonuçları sakla
    all_models_results = {}
    all_models_metrics = []

    # En iyi hibrit modeli DenseNet169 (Standart)'dan ayrı olarak izle
    best_hybrid_model_name = None
    best_hybrid_model_f1 = -1
    best_hybrid_model_predictions = None
    best_hybrid_model_probabilities = None
    best_hybrid_model_true_labels = None

    # 1. Standart DenseNet169 modelini eğit
    print("\n===== Training Standard DenseNet169 Model =====")
    model = build_densenet_model(num_classes)
    model, history = fast_train(model, train_generator, validation_generator)

    # Eğitim geçmişini çizdir
    plot_training_history(history)

    # Modeli değerlendir
    densenet_metrics, y_true, y_pred, y_pred_prob = evaluate_densenet(model, test_generator)
    densenet_metrics['model'] = "DenseNet169 (Standard)"

    # DenseNet için sınıf başına metrikleri hesapla
    densenet_per_class_df = calculate_per_class_metrics(y_true, y_pred, class_names)

    # Sonuçlara ekle
    all_models_metrics.append(densenet_metrics)
    all_models_results["DenseNet169 (Standard)"] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_pred_prob
    }

    # DenseNet169 için karışıklık matrisini çizdir
    plot_confusion_matrix(y_true, y_pred, class_names, "DenseNet169 Confusion Matrix")

    # DenseNet169 performansını sakla - ancak hibrit modellerle karşılaştırma
    densenet_f1 = densenet_metrics['f1']

    # 2. Özellik çıkarıcı oluştur ve özellikleri çıkar
    print("\n===== Extracting Features from DenseNet169 =====")
    feature_extractor = create_feature_extractor()

    # Üreteçleri başlangıca sıfırla
    train_generator.reset()
    validation_generator.reset()
    test_generator.reset()

    # Özellikleri çıkar
    print("Extracting training features...")
    train_features, train_labels = extract_features(feature_extractor, train_generator)
    print(f"Training features shape: {train_features.shape}")

    print("Extracting validation features...")
    val_features, val_labels = extract_features(feature_extractor, validation_generator)
    print(f"Validation features shape: {val_features.shape}")

    print("Extracting test features...")
    test_features, test_labels = extract_features(feature_extractor, test_generator)
    print(f"Test features shape: {test_features.shape}")

    # 3. Farklı çekirdeklerle SVM eğit ve değerlendir
    print("\n===== Training and Evaluating SVMs with Different Kernels =====")

    # Doğrusal çekirdekli SVM
    svm_linear_metrics, svm_linear_pred, svm_linear_prob = train_and_evaluate_svm(
        train_features, train_labels, test_features, test_labels, kernel='linear'
    )
    all_models_metrics.append(svm_linear_metrics)
    all_models_results["DenseNet169 + SVM (linear)"] = {
        "y_true": test_labels,
        "y_pred": svm_linear_pred,
        "y_prob": svm_linear_prob
    }

    # Şimdiye kadarki en iyi hibrit model olup olmadığını kontrol et
    if svm_linear_metrics['f1'] > best_hybrid_model_f1:
        best_hybrid_model_f1 = svm_linear_metrics['f1']
        best_hybrid_model_name = "DenseNet169 + SVM (linear)"
        best_hybrid_model_predictions = svm_linear_pred
        best_hybrid_model_probabilities = svm_linear_prob
        best_hybrid_model_true_labels = test_labels

    # RBF çekirdekli SVM
    svm_rbf_metrics, svm_rbf_pred, svm_rbf_prob = train_and_evaluate_svm(
        train_features, train_labels, test_features, test_labels, kernel='rbf'
    )
    all_models_metrics.append(svm_rbf_metrics)
    all_models_results["DenseNet169 + SVM (RBF)"] = {
        "y_true": test_labels,
        "y_pred": svm_rbf_pred,
        "y_prob": svm_rbf_prob
    }

    # Şimdiye kadarki en iyi hibrit model olup olmadığını kontrol et
    if svm_rbf_metrics['f1'] > best_hybrid_model_f1:
        best_hybrid_model_f1 = svm_rbf_metrics['f1']
        best_hybrid_model_name = "DenseNet169 + SVM (RBF)"
        best_hybrid_model_predictions = svm_rbf_pred
        best_hybrid_model_probabilities = svm_rbf_prob
        best_hybrid_model_true_labels = test_labels

    # Polinom çekirdekli SVM
    svm_poly_metrics, svm_poly_pred, svm_poly_prob = train_and_evaluate_svm(
        train_features, train_labels, test_features, test_labels, kernel='poly'
    )
    all_models_metrics.append(svm_poly_metrics)
    all_models_results["DenseNet169 + SVM (Poly)"] = {
        "y_true": test_labels,
        "y_pred": svm_poly_pred,
        "y_prob": svm_poly_prob
    }

    # Şimdiye kadarki en iyi hibrit model olup olmadığını kontrol et
    if svm_poly_metrics['f1'] > best_hybrid_model_f1:
        best_hybrid_model_f1 = svm_poly_metrics['f1']
        best_hybrid_model_name = "DenseNet169 + SVM (Poly)"
        best_hybrid_model_predictions = svm_poly_pred
        best_hybrid_model_probabilities = svm_poly_prob
        best_hybrid_model_true_labels = test_labels

    # Sigmoid çekirdekli SVM
    svm_sigmoid_metrics, svm_sigmoid_pred, svm_sigmoid_prob = train_and_evaluate_svm(
        train_features, train_labels, test_features, test_labels, kernel='sigmoid'
    )
    all_models_metrics.append(svm_sigmoid_metrics)
    all_models_results["DenseNet169 + SVM (Sigmoid)"] = {
        "y_true": test_labels,
        "y_pred": svm_sigmoid_pred,
        "y_prob": svm_sigmoid_prob
    }

    # Şimdiye kadarki en iyi hibrit model olup olmadığını kontrol et
    # Şimdiye kadarki en iyi hibrit model olup olmadığını kontrol et
    if svm_sigmoid_metrics['f1'] > best_hybrid_model_f1:
        best_hybrid_model_f1 = svm_sigmoid_metrics['f1']
        best_hybrid_model_name = "DenseNet169 + SVM (Sigmoid)"
        best_hybrid_model_predictions = svm_sigmoid_pred
        best_hybrid_model_probabilities = svm_sigmoid_prob
        best_hybrid_model_true_labels = test_labels

    # 4. Boosting yöntemlerini eğit ve değerlendir
    print("\n===== Training and Evaluating Boosting Methods =====")

    # Gradient Boosting
    gb_metrics, gb_pred, gb_prob = train_and_evaluate_gradient_boosting(
        train_features, train_labels, test_features, test_labels
    )
    all_models_metrics.append(gb_metrics)
    all_models_results["DenseNet169 + Gradient Boosting"] = {
        "y_true": test_labels,
        "y_pred": gb_pred,
        "y_prob": gb_prob
    }

    # Şimdiye kadarki en iyi hibrit model olup olmadığını kontrol et
    if gb_metrics['f1'] > best_hybrid_model_f1:
        best_hybrid_model_f1 = gb_metrics['f1']
        best_hybrid_model_name = "DenseNet169 + Gradient Boosting"
        best_hybrid_model_predictions = gb_pred
        best_hybrid_model_probabilities = gb_prob
        best_hybrid_model_true_labels = test_labels

    # XGBoost (mevcutsa)
    if XGBOOST_AVAILABLE:
        xgb_metrics, xgb_pred, xgb_prob = train_and_evaluate_xgboost(
            train_features, train_labels, test_features, test_labels
        )
        all_models_metrics.append(xgb_metrics)
        all_models_results["DenseNet169 + XGBoost"] = {
            "y_true": test_labels,
            "y_pred": xgb_pred,
            "y_prob": xgb_prob
        }

        # Şimdiye kadarki en iyi hibrit model olup olmadığını kontrol et
        if xgb_metrics['f1'] > best_hybrid_model_f1:
            best_hybrid_model_f1 = xgb_metrics['f1']
            best_hybrid_model_name = "DenseNet169 + XGBoost"
            best_hybrid_model_predictions = xgb_pred
            best_hybrid_model_probabilities = xgb_prob
            best_hybrid_model_true_labels = test_labels
    else:
        print("Skipping XGBoost (not installed)")

    # LightGBM (mevcutsa)
    if LIGHTGBM_AVAILABLE:
        lgbm_metrics, lgbm_pred, lgbm_prob = train_and_evaluate_lightgbm(
            train_features, train_labels, test_features, test_labels
        )
        all_models_metrics.append(lgbm_metrics)
        all_models_results["DenseNet169 + LightGBM"] = {
            "y_true": test_labels,
            "y_pred": lgbm_pred,
            "y_prob": lgbm_prob
        }

        # Şimdiye kadarki en iyi hibrit model olup olmadığını kontrol et
        if lgbm_metrics['f1'] > best_hybrid_model_f1:
            best_hybrid_model_f1 = lgbm_metrics['f1']
            best_hybrid_model_name = "DenseNet169 + LightGBM"
            best_hybrid_model_predictions = lgbm_pred
            best_hybrid_model_probabilities = lgbm_prob
            best_hybrid_model_true_labels = test_labels
    else:
        print("Skipping LightGBM (not installed)")

    # CatBoost (mevcutsa)
    if CATBOOST_AVAILABLE:
        catboost_metrics, catboost_pred, catboost_prob = train_and_evaluate_catboost(
            train_features, train_labels, test_features, test_labels
        )
        all_models_metrics.append(catboost_metrics)

        # Sadece geçerli tahminlere sahipsek ROC karşılaştırmasına ekle
        if catboost_prob is not None:
            all_models_results["DenseNet169 + CatBoost"] = {
                "y_true": test_labels,
                "y_pred": catboost_pred,
                "y_prob": catboost_prob
            }

            # Şimdiye kadarki en iyi hibrit model olup olmadığını kontrol et
            if catboost_metrics['f1'] > best_hybrid_model_f1:
                best_hybrid_model_f1 = catboost_metrics['f1']
                best_hybrid_model_name = "DenseNet169 + CatBoost"
                best_hybrid_model_predictions = catboost_pred
                best_hybrid_model_probabilities = catboost_prob
                best_hybrid_model_true_labels = test_labels
    else:
        print("Skipping CatBoost (not installed or incompatible)")

    # AdaBoost
    ada_metrics, ada_pred, ada_prob = train_and_evaluate_adaboost(
        train_features, train_labels, test_features, test_labels
    )
    all_models_metrics.append(ada_metrics)
    all_models_results["DenseNet169 + AdaBoost"] = {
        "y_true": test_labels,
        "y_pred": ada_pred,
        "y_prob": ada_prob
    }

    # Şimdiye kadarki en iyi hibrit model olup olmadığını kontrol et
    if ada_metrics['f1'] > best_hybrid_model_f1:
        best_hybrid_model_f1 = ada_metrics['f1']
        best_hybrid_model_name = "DenseNet169 + AdaBoost"
        best_hybrid_model_predictions = ada_pred
        best_hybrid_model_probabilities = ada_prob
        best_hybrid_model_true_labels = test_labels

    # 5. Tüm modelleri karşılaştır
    print("\n===== Comparing All Models =====")

    # Şimdi DenseNet169 (Standart)'dan ayrı bir hibrit modele sahip olduğumuzdan eminiz
    if best_hybrid_model_name:
        print(f"Best hybrid model: {best_hybrid_model_name} with F1 score: {best_hybrid_model_f1:.4f}")
        # Temel ile karşılaştır
        if densenet_f1 > best_hybrid_model_f1:
            print(f"Note: DenseNet169 (Standard) still outperforms hybrid models with F1 score: {densenet_f1:.4f}")
        else:
            print(f"Hybrid model outperforms DenseNet169 (Standard) (F1: {densenet_f1:.4f})")
    else:
        print("No viable hybrid models found.")

    # Tüm metriklerle veri çerçevesi oluştur
    all_metrics_df = pd.DataFrame(all_models_metrics)

    # En iyi hibrit model için sınıf başına metrikler oluştur
    best_hybrid_per_class_df = calculate_per_class_metrics(
        best_hybrid_model_true_labels, best_hybrid_model_predictions, class_names
    )

    # Metrik tablolarını oluştur ve kaydet
    create_metrics_tables(densenet_per_class_df, best_hybrid_per_class_df)

    # Metrikleri 4 ondalık basamağa biçimlendir
    for col in ['accuracy', 'precision', 'recall', 'f1', 'f2', 'f0', 'specificity', 'mcc', 'kappa']:
        all_metrics_df[col] = all_metrics_df[col].map(lambda x: f"{x:.4f}")

    # Metrikleri görüntüle
    print("\nMetrics for all models:")
    print(all_metrics_df)

    # CSV'ye kaydet
    all_metrics_df.to_csv('all_models_metrics.csv', index=False)
    print("Metrics saved to 'all_models_metrics.csv'")

    # 6. Metrikleri görselleştir
    print("\n===== Visualizing Metrics =====")

    # Görselleştirme için dize metriklerini tekrar float'a dönüştür
    for col in ['accuracy', 'precision', 'recall', 'f1', 'f2', 'f0', 'specificity', 'mcc', 'kappa']:
        all_metrics_df[col] = all_metrics_df[col].astype(float)

    # Görselleştirmeler oluştur
    visualize_metrics(all_metrics_df)

    # 7. En iyi hibrit model için karışıklık matrisini çizdir
    plot_confusion_matrix(
        best_hybrid_model_true_labels,
        best_hybrid_model_predictions,
        class_names,
        f"{best_hybrid_model_name} Confusion Matrix"
    )

    # 8. ROC eğrilerini hesapla ve çizdir
    print("\n===== Calculating and Plotting ROC Curves =====")
    roc_df, roc_aucs = calculate_and_plot_roc(all_models_results, class_names, num_classes)

    # 9. En iyi hibrit model için hassasiyet-geri çağırma eğrisini çizdir
    print("\n===== Calculating and Plotting Precision-Recall Curve for Best Hybrid Model =====")
    average_precision = plot_precision_recall_curve(
        best_hybrid_model_true_labels,
        best_hybrid_model_probabilities,
        class_names,
        num_classes,
        best_hybrid_model_name
    )

    print("\n===== Enhanced Classification Complete =====")

    return {
        'all_metrics_df': all_metrics_df,
        'roc_df': roc_df,
        'densenet_per_class_df': densenet_per_class_df,
        'best_hybrid_per_class_df': best_hybrid_per_class_df,
        'best_hybrid_model_name': best_hybrid_model_name,
        'best_hybrid_model_f1': best_hybrid_model_f1,
        'densenet_f1': densenet_f1
    }
```

## Kullanım Örneği

Kodun nasıl kullanılacağını gösteren ana yürütme bloğu:

```python
# Kullanım örneği:
if __name__ == "__main__":
    # Veri kümenizin yoluyla değiştirin
    DATA_FOLDER = "/content/drive/MyDrive/train"

    # Gelişmiş sınıflandırmayı çalıştırın
    results = run_enhanced_classification(DATA_FOLDER)

    if results:
        print(f"\nBest hybrid model: {results['best_hybrid_model_name']} with F1 score: {results['best_hybrid_model_f1']:.4f}")
        print(f"DenseNet169 (Standard) F1 score: {results['densenet_f1']:.4f}")

        # Özet rapor oluştur
        print("\n===== Summary Report =====")
        print("1. DenseNet169 Per-Class Performance:")
        print(results['densenet_per_class_df'][['class', 'accuracy', 'precision', 'recall', 'f1']])

        print("\n2. Best Hybrid Model Per-Class Performance:")
        print(results['best_hybrid_per_class_df'][['class', 'accuracy', 'precision', 'recall', 'f1']])

        print("\n3. Top 3 Models by ROC AUC:")
        print(results['roc_df'].head(3))
```

## Kodun İşleyişi: Adım Adım Açıklama

Bu bölümde, sistemin tam olarak nasıl çalıştığını adım adım açıklayacağız.

### 1. Başlangıç Kurulumu

Kod, öncelikle gerekli tüm kütüphaneleri içe aktarır ve belirli kütüphanelerin kullanılabilirliğini kontrol eder (XGBoost, LightGBM, CatBoost). Bu, farklı sistemlerde çalışabilirlik sağlar; bazı gelişmiş kütüphaneler yüklü değilse, kod bu sınıflandırıcıları atlayarak devam eder.

Sabit değerler (IMAGE_SIZE, BATCH_SIZE, EPOCHS, vb.) bu aşamada tanımlanır. Bu, model konfigürasyonunun merkezi bir yerden kontrol edilmesini sağlar.

### 2. Veri Yükleme ve Ön İşleme

`run_enhanced_classification` fonksiyonu, veri klasörünün yapısını kontrol eder ve uygun şekilde üreteçler oluşturur. Kod iki tür veri organizasyonunu destekler:
- Sınıf alt klasörlerine sahip tek bir klasör
- Ayrı eğitim ve test klasörleri

Veriler, standart görüntü yeniden ölçeklendirme (1/255) ve veri artırma teknikleri (döndürme, kaydırma, yatay çevirme) kullanılarak yüklenir. Bu, modelin farklı varyasyonlara daha dayanıklı olmasını sağlar.

### 3. Temel DenseNet169 Modelinin Eğitimi

İlk adım olarak, kod standart bir DenseNet169 modelini eğitir. Model mimarisi, `build_densenet_model` fonksiyonunda tanımlanır ve şunları içerir:
- ImageNet ağırlıklarıyla önceden eğitilmiş DenseNet169 temel modeli
- Dondurulmuş temel model katmanları (transfer öğrenimi)
- 1024 nöronlu bir yoğun (dense) katman
- 0.5 dropout oranı
- Sınıf sayısına uygun bir çıkış katmanı

Model, `fast_train` fonksiyonu kullanılarak eğitilir, bu fonksiyon erken durdurma ile doğrulama kaybını izler. Bu, modelin aşırı uyumunu (overfitting) önler.

### 4. Özellik Çıkarma

Temel DenseNet169 modeli eğitildikten sonra, kod eğitim, doğrulama ve test verileri için özellikler çıkarmak için ayrı bir özellik çıkarıcı oluşturur. Bu, DenseNet169'i yalnızca özellik çıkarıcı olarak kullanmak için özellikle oluşturulmuş farklı bir modeldir.

Çıkarılan özellikler daha sonra çeşitli geleneksel makine öğrenimi sınıflandırıcılarının eğitimi için kullanılır.

### 5. Hibrit Modellerin Eğitimi

Kod, çıkarılan özellikler üzerinde aşağıdaki sınıflandırıcıları eğitir:
- SVM: Farklı çekirdek fonksiyonlarıyla (linear, rbf, poly, sigmoid)
- Gradient Boosting
- XGBoost (mevcutsa)
- LightGBM (mevcutsa)
- CatBoost (mevcutsa)
- AdaBoost

Her model, tüm kategorilerin ağırlıklı ortalama metriklerini hesaplayan `train_and_evaluate_*` fonksiyonları kullanılarak değerlendirilir.

### 6. En İyi Modelin Seçimi

Eğitim ve değerlendirme sırasında, kod sürekli olarak en iyi F1 skoruna sahip hibrit modeli takip eder. F1 skoru, kesinlik ve duyarlılık arasında bir denge sağladığı için tercih edilir.

Son değerlendirmede, en iyi hibrit model standart DenseNet169 ile karşılaştırılır. Bazen, özellikle veri kümesi küçükse, standart DenseNet169 hibrit modellerden daha iyi performans gösterebilir.

### 7. Sonuçların Görselleştirilmesi ve Raporlanması

Kod, eğitim ve değerlendirme sonuçlarını çeşitli görselleştirmelerle kapsamlı bir şekilde raporlar:
- Eğitim geçmişi grafikleri (doğruluk ve kayıp)
- Karışıklık matrisleri
- ROC eğrileri
- Hassasiyet-geri çağırma eğrileri
- Metrik karşılaştırma grafikleri
- Sınıf başına performans tabloları

Tüm sonuçlar, daha sonra inceleme için CSV dosyalarına ve yüksek çözünürlüklü görüntülere kaydedilir.

## Kod Mimarisinin Güçlü Yönleri

1. **Modülerlik**: Kod, farklı fonksiyonlara bölünmüştür, bu da bakımı ve anlaşılmasını kolaylaştırır.

2. **Esneklik**: Sistem farklı veri klasörü yapılarını destekler ve isteğe bağlı kütüphanelerin olmadığı durumları zarif bir şekilde ele alır.

3. **Kapsamlı Değerlendirme**: Doğruluk, kesinlik, duyarlılık, F1, özgüllük, MCC ve diğerleri dahil olmak üzere birçok metrik hesaplanır.

4. **Görsel Analiz**: Sonuçları yorumlamayı kolaylaştıran zengin görselleştirmeler.

5. **Hibrit Yaklaşım**: Derin öğrenme ve geleneksel makine öğrenimi yöntemlerinin güçlü yönlerini birleştirir.

## Optimize Edilebilecek Alanlar

1. **Hiperparametre Ayarı**: Sınıflandırıcılar için hiperparametre optimizasyonu (örn. grid search) eklenebilir.

2. **Özellik Seçimi**: Önemli özellikleri seçmek için özellik seçim yöntemleri eklenebilir.

3. **Derin Özellik Ayarı**: DenseNet169'in son birkaç katmanını ince ayarlamak için kod genişletilebilir.

4. **Topluluk Yöntemleri**: Birden fazla modelin tahminlerini birleştirmek için topluluk yöntemleri eklenebilir.

5. **Veri Dengesizliği**: Dengesiz veri kümeleri için sınıf ağırlıkları veya örnekleme stratejileri eklenebilir.

Bu kod, özellikle görüntü sınıflandırması için sağlam, kapsamlı ve genişletilebilir bir çerçeve sağlar. Transfer öğrenimi ilkeleri uygulanarak, derin öğrenme özellik çıkarma yetenekleri geleneksel makine öğrenimi yöntemleriyle verimli bir şekilde birleştirilir.# DenseNet169 Hibrit Sınıflandırma Kodu Detaylı Açıklaması
