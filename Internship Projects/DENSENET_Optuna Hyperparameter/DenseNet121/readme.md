# DENSENET121_Optuna Hyperparameter

---

Bu detaylı dokümantasyon, DenseNet121 tabanlı görüntü sınıflandırma sisteminin tüm yönlerini kapsamaktadır. Kod'un her bölümü, kullanım senaryoları ve optimizasyon stratejileri açık bir şekilde açıklanmıştır.# DenseNet121 ile Gelişmiş Görüntü Sınıflandırma - Detaylı Kod Analizi

Bu projede, DenseNet121 önceden eğitilmiş modelini kullanarak kapsamlı bir görüntü sınıflandırma sistemi geliştirilmiştir. Bu doküman, kodun her bölümünü detaylı olarak açıklamaktadır.

## 📋 İçerik Tablosu

1. [Proje Genel Bakış](#proje-genel-bakış)
2. [Kütüphane İmportları ve Kurulumlar](#kütüphane-i̇mportları-ve-kurulumlar)
3. [Sabit Parametreler](#sabit-parametreler)
4. [Metrik Hesaplama Fonksiyonları](#metrik-hesaplama-fonksiyonları)
5. [DenseNet121 Model Fonksiyonları](#densenet121-model-fonksiyonları)
6. [Makine Öğrenmesi Algoritmaları](#makine-öğrenmesi-algoritmaları)
7. [Görselleştirme Fonksiyonları](#görselleştirme-fonksiyonları)
8. [Optuna Optimizasyon Fonksiyonları](#optuna-optimizasyon-fonksiyonları)
9. [Ana Uygulama Fonksiyonu](#ana-uygulama-fonksiyonu)
10. [Kullanım Örnekleri](#kullanım-örnekleri)

## 1. Proje Genel Bakış

### 1.1 Sistem Mimarisi

```
[Görüntü Verisi] → [DenseNet121 Feature Extractor] → [ML Algoritmaları] → [Tahmin]
                          ↓
                    [Hiperparametre Optimizasyonu (Optuna)]
                          ↓
                    [Performans Metrikleri]
```

### 1.2 Ana Akış

1. **Veri Hazırlama**: ImageDataGenerator ile veri augmentasyonu
2. **DenseNet121 Eğitimi**: Transfer learning ile fine-tuning
3. **Özellik Çıkarma**: Eğitilen modelden feature extraction
4. **Hibrit Modeller**: ML algoritmaları ile sınıflandırma
5. **Optimizasyon**: Optuna ile hiperparametre ayarlama
6. **Değerlendirme**: 9 farklı metrik ile analiz

## 2. Kütüphane İmportları ve Kurulumlar

### 2.1 Temel Kütüphaneler

```python
import os                    # Dosya sistemi operasyonları
import numpy as np          # Numerical computing
import pandas as pd         # Veri manipülasyonu
import matplotlib.pyplot as plt   # Görselleştirme
import seaborn as sns       # Gelişmiş plotting
```

### 2.2 TensorFlow/Keras İmportları

```python
from tensorflow.keras.applications import DenseNet121  # Önceden eğitilmiş model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Veri yükleme
from tensorflow.keras.models import Model  # Model oluşturma
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout  # Katmanlar
from tensorflow.keras.optimizers import RMSprop, Adam, SGD  # Optimizerlar
from tensorflow.keras.callbacks import EarlyStopping  # Erken durdurma
```

### 2.3 Opsiyonel Kütüphane Kontrolü

```python
# LightGBM için try-except bloğu
try:
    import lightgbm as lgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM is not installed. Skipping LightGBM classifier.")
```

**Açıklama**: Bu yapı, kütüphanenin eksik olması durumunda programın çökmesini önler ve kullanıcıyı bilgilendirir.

## 3. Sabit Parametreler

```python
IMAGE_SIZE = (150, 150)     # Görüntü boyutu
BATCH_SIZE = 32             # Batch boyutu
EPOCHS = 30                 # Eğitim epoch sayısı
LEARNING_RATE = 0.0001      # Öğrenme oranı
VAL_SPLIT = 0.2             # Doğrulama veri oranı
```

**Açıklama**: Bu parametreler projenin temel konfigürasyonunu oluşturur ve kolayca değiştirilebilir.

## 4. Metrik Hesaplama Fonksiyonları

### 4.1 Özel Specificity Metriği

```python
def calculate_specificity(y_true, y_pred):
    """Calculate specificity for multi-class classification
    
    Specificity = TN / (TN + FP)
    """
    cm = confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)  # False Positives
    tn = cm.sum() - (fp + np.diag(cm) + cm.sum(axis=1) - np.diag(cm))  # True Negatives
    
    specificity = np.zeros_like(tn, dtype=float)
    for i in range(len(specificity)):
        if tn[i] + fp[i] > 0:
            specificity[i] = tn[i] / (tn[i] + fp[i])
        else:
            specificity[i] = 0.0
    
    # Ağırlıklı ortalama döndür
    return np.average(specificity, weights=np.bincount(y_true) if len(np.unique(y_true)) > 1 else None)
```

**Detaylı Açıklama**:
- Confusion matrix'ten true negative ve false positive değerlerini hesaplar
- Her sınıf için specificity hesaplar
- Class imbalance durumunda ağırlıklı ortalama alır

### 4.2 Genel Metrik Hesaplama

```python
def calculate_metrics(y_true, y_pred):
    """Calculate all required metrics"""
    metrics = {}
    
    # Temel metrikler
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    
    # F-skorları (beta parametreleri ile)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['f2'] = fbeta_score(y_true, y_pred, beta=2, average='weighted')
    metrics['f0'] = fbeta_score(y_true, y_pred, beta=0.5, average='weighted')
    
    # Gelişmiş metrikler
    metrics['specificity'] = calculate_specificity(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    
    return metrics
```

**F-Beta Skoru Açıklaması**:
- **F1 (β=1)**: Precision ve Recall'u eşit ağırlıkta değerlendirir
- **F2 (β=2)**: Recall'a daha fazla önem verir
- **F0.5 (β=0.5)**: Precision'a daha fazla önem verir

### 4.3 Sınıf Bazlı Metrik Hesaplama

```python
def calculate_per_class_metrics(y_true, y_pred, class_names):
    """Calculate metrics for each class"""
    n_classes = len(class_names)
    
    # Sonuç dictionary'si başlat
    results = {
        'class': class_names,
        'accuracy': [], 'precision': [], 'recall': [],
        'specificity': [], 'f0': [], 'f1': [], 'f2': [],
        'kappa': [], 'mcc': []
    }
    
    # Confusion matrix hesapla
    cm = confusion_matrix(y_true, y_pred)
    
    # Her sınıf için metrikleri hesapla
    for i in range(n_classes):
        # Binary classification metrikleri için TP, FP, FN, TN hesapla
        tp = cm[i, i]  # True positives
        fp = np.sum(cm[:, i]) - tp  # False positives
        fn = np.sum(cm[i, :]) - tp  # False negatives
        tn = np.sum(cm) - tp - fp - fn  # True negatives
        
        # Metrik hesaplamaları...
```

**Açıklama**: Her sınıf için ayrı ayrı One-vs-Rest yaklaşımı ile binary classification metrikleri hesaplar.

## 5. DenseNet121 Model Fonksiyonları

### 5.1 Feature Extractor Oluşturma

```python
def create_feature_extractor():
    # DenseNet121'i ImageNet ağırlıkları ile yükle
    base_model = DenseNet121(
        weights='imagenet',        # Önceden eğitilmiş ağırlıklar
        include_top=False,         # Classification katmanını dahil etme
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)  # RGB görüntü boyutu
    )
    
    # Tüm katmanları dondur (transfer learning)
    for layer in base_model.layers:
        layer.trainable = False
    
    # Global average pooling ekle
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Feature extractor model oluştur
    model = Model(inputs=base_model.input, outputs=x)
    
    return model
```

**Açıklama**:
- **include_top=False**: Son classification katmanını çıkarır
- **weights='imagenet'**: ImageNet üzerinde eğitilmiş ağırlıkları kullanır
- **GlobalAveragePooling2D**: Feature map'leri vektöre dönüştürür

### 5.2 Feature Extraction İşlemi

```python
def extract_features(feature_extractor, data_generator):
    features = []
    labels = []
    
    # Tüm batch'ler için feature extraction
    num_batches = len(data_generator)
    for i in range(num_batches):
        x_batch, y_batch = data_generator[i]  # Batch al
        batch_features = feature_extractor.predict(x_batch, verbose=0)  # Feature çıkar
        features.append(batch_features)
        labels.append(np.argmax(y_batch, axis=1))  # One-hot'tan label'a çevir
    
    # Tüm batch'leri birleştir
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return features, labels
```

### 5.3 DenseNet121 Model Oluşturma

```python
def build_densenet_model(num_classes, dense_neurons=1024, dropout_rate=0.5,
                         optimizer='rmsprop', learning_rate=LEARNING_RATE):
    # Base model yükle
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    
    # Base model katmanlarını dondur
    for layer in base_model.layers:
        layer.trainable = False
    
    # Classification katmanları ekle
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_neurons, activation='relu')(x)  # Özelleştirilebilir dense layer
    x = Dropout(dropout_rate)(x)                    # Overfitting önleme
    predictions = Dense(num_classes, activation='softmax')(x)  # Çıkış katmanı
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Optimizer seçimi
    optimizer_map = {
        'rmsprop': RMSprop(learning_rate=learning_rate),
        'adam': Adam(learning_rate=learning_rate),
        'sgd': SGD(learning_rate=learning_rate)
    }
    opt = optimizer_map.get(optimizer, RMSprop(learning_rate=learning_rate))
    
    # Model compile
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

**Katman Açıklamaları**:
- **GlobalAveragePooling2D**: 7x7x1024 → 1024 boyutunda vektör
- **Dense(dense_neurons)**: Fully connected layer (özelleştirilebilir)
- **Dropout**: Regularization tekniği
- **Dense(num_classes, softmax)**: Son sınıflandırma katmanı

## 6. Makine Öğrenmesi Algoritmaları

### 6.1 SVM Implementation

```python
def train_and_evaluate_svm(train_features, train_labels, test_features, test_labels,
                          kernel='linear', C=1.0, gamma='scale'):
    print(f"Training SVM with {kernel} kernel...")
    
    # SVM model oluştur
    svm = SVC(
        kernel=kernel,          # Kernel türü: linear, rbf, poly, sigmoid
        C=C,                   # Regularization parametresi
        gamma=gamma,           # Kernel coefficient
        probability=True,      # Probability prediction için
        random_state=42        # Reproducibility için
    )
    
    # Model eğit
    svm.fit(train_features, train_labels)
    
    # Tahminler
    y_pred = svm.predict(test_features)
    y_pred_prob = svm.predict_proba(test_features)
    
    # Metrikleri hesapla
    metrics = calculate_metrics(test_labels, y_pred)
    metrics['model'] = f"DenseNet121 + SVM ({kernel})"
    
    return metrics, y_pred, y_pred_prob, svm
```

**SVM Kernel Açıklamaları**:
- **Linear**: Doğrusal ayrılabilir veriler için
- **RBF (Radial Basis Function)**: Non-linear veriler için yaygın
- **Polynomial**: Belirli derecelerde polynomial features
- **Sigmoid**: Neural network benzeri aktivasyon

### 6.2 Gradient Boosting Implementation

```python
def train_and_evaluate_gradient_boosting(train_features, train_labels, test_features, test_labels,
                                         n_estimators=100, learning_rate=0.1, max_depth=3):
    print("Training Gradient Boosting classifier...")
    
    # Gradient Boosting model oluştur
    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,    # Ağaç sayısı
        learning_rate=learning_rate,  # Shrinkage parametresi
        max_depth=max_depth,          # Ağaç derinliği
        random_state=42
    )
    
    # Model eğit ve değerlendir
    gb.fit(train_features, train_labels)
    y_pred = gb.predict(test_features)
    y_pred_prob = gb.predict_proba(test_features)
    
    metrics = calculate_metrics(test_labels, y_pred)
    metrics['model'] = "DenseNet121 + Gradient Boosting"
    
    return metrics, y_pred, y_pred_prob, gb
```

### 6.3 XGBoost Implementation

```python
def train_and_evaluate_xgboost(train_features, train_labels, test_features, test_labels,
                              n_estimators=100, learning_rate=0.1, max_depth=6):
    print("Training XGBoost classifier...")
    
    # XGBoost parametreleri
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=1,        # Minimum weight için leaf node
        gamma=0,                   # Minimum split loss
        subsample=0.8,             # Row sampling
        colsample_bytree=0.8,      # Column sampling
        objective='multi:softprob', # Multi-class probability
        num_class=len(np.unique(train_labels)),
        random_state=42,
        eval_metric='mlogloss'     # Evaluation metric
    )
    
    # Training ve prediction
    xgb_classifier.fit(train_features, train_labels)
    y_pred = xgb_classifier.predict(test_features)
    y_pred_prob = xgb_classifier.predict_proba(test_features)
    
    metrics = calculate_metrics(test_labels, y_pred)
    metrics['model'] = "DenseNet121 + XGBoost"
    
    return metrics, y_pred, y_pred_prob, xgb_classifier
```

**XGBoost Parametre Açıklamaları**:
- **subsample**: Her iteration'da kullanılacak sample oranı
- **colsample_bytree**: Her tree'de kullanılacak feature oranı
- **gamma**: Leaf node split için minimum gain
- **min_child_weight**: Child node için minimum weight sum

## 7. Görselleştirme Fonksiyonları

### 7.1 Training History Plotting

```python
def plot_training_history(history):
    """Plot training and validation accuracy and loss curves"""
    # 2 subplot oluştur
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('DenseNet121 Model Accuracy', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.legend(['Train', 'Validation'], loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('DenseNet121 Model Loss', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(['Train', 'Validation'], loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('densenet121_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig
```

### 7.2 Confusion Matrix Visualization

```python
def plot_confusion_matrix(y_true, y_pred, class_names, title):
    """Plot confusion matrix with custom styling"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, 
                annot=True,            # Hücrelerde sayıları göster
                fmt='d',               # Integer format
                cmap='Blues',          # Color map
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Dosya kaydet
    filename = title.replace(' ', '_').lower() + '.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm
```

### 7.3 ROC Curve Analysis

```python
def calculate_and_plot_roc(all_models_results, class_names, num_classes):
    plt.figure(figsize=(15, 12))
    
    # ROC AUC değerlerini sakla
    roc_aucs = {}
    
    # Her model için
    for model_name, model_data in all_models_results.items():
        y_true = model_data['y_true']
        y_prob = model_data['y_prob']
        
        # ROC curve ve AUC hesapla
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # One-hot encoding
        y_true_bin = np.zeros((len(y_true), num_classes))
        for i in range(len(y_true)):
            y_true_bin[i, y_true[i]] = 1
        
        # Her sınıf için ROC
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Micro-average ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # ROC AUC'u kaydet
        roc_aucs[model_name] = roc_auc["micro"]
        
        # ROC eğrisini çiz
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'{model_name} (AUC = {roc_auc["micro"]:.4f})',
                 linewidth=2)
    
    # Diagonal çiz (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    
    # Plot ayarları
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves for All Models', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return roc_aucs
```

## 8. Optuna Optimizasyon Fonksiyonları

### 8.1 DenseNet Objective Function

```python
def objective_densenet(trial, data_folder):
    """Optuna objective function for optimizing DenseNet121 hyperparameters"""
    
    # Optimizasyon için hiperparametreler
    dense_neurons = trial.suggest_categorical('dense_neurons', [512, 1024, 2048])
    dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.7)
    optimizer_name = trial.suggest_categorical('optimizer', ['rmsprop', 'adam', 'sgd'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    
    # Data augmentation parametreleri
    aug_rotation = trial.suggest_int('aug_rotation', 5, 20)
    aug_width_shift = trial.suggest_float('aug_width_shift', 0.05, 0.2)
    aug_height_shift = trial.suggest_float('aug_height_shift', 0.05, 0.2)
    aug_zoom = trial.suggest_float('aug_zoom', 0.05, 0.2)
    
    # Veri generatorları oluştur
    train_generator, validation_generator, _ = create_data_generators(
        data_folder,
        batch_size=batch_size,
        aug_rotation=aug_rotation,
        aug_width_shift=aug_width_shift,
        aug_height_shift=aug_height_shift,
        aug_zoom=aug_zoom
    )
    
    # Model oluştur
    num_classes = len(train_generator.class_indices)
    model = build_densenet_model(
        num_classes=num_classes,
        dense_neurons=dense_neurons,
        dropout_rate=dropout_rate,
        optimizer=optimizer_name,
        learning_rate=learning_rate
    )
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Optuna pruning callback
    pruning_callback = TFKerasPruningCallback(trial, 'val_accuracy')
    
    # Model eğit
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=15,  # Hızlı denemeler için azaltılmış
        callbacks=[early_stop, pruning_callback],
        verbose=1
    )
    
    # Validation accuracy'si döndür
    return history.history['val_accuracy'][-1]
```

**Optuna Açıklamaları**:
- **suggest_categorical**: Belirli kategorilerden seçim
- **suggest_float**: Sürekli değer aralığı (log=True için logaritmik)
- **suggest_int**: Integer aralık
- **TFKerasPruningCallback**: Başarısız denemeleri erken sonlandırır

### 8.2 SVM Objective Function

```python
def objective_svm(trial, train_features, train_labels, val_features, val_labels):
    """Optuna objective function for optimizing SVM hyperparameters"""
    
    # Hiperparametre seçimi
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    C = trial.suggest_float('C', 0.1, 10.0, log=True)
    
    # Linear olmayan kerneller için gamma
    if kernel != 'linear':
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    else:
        gamma = 'scale'
    
    # Polynomial kernel için ek parametreler
    if kernel == 'poly':
        degree = trial.suggest_int('degree', 2, 5)
        coef0 = trial.suggest_float('coef0', 0.0, 1.0)
        svm = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, 
                  coef0=coef0, probability=True, random_state=42)
    else:
        svm = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
    
    # Model eğit ve değerlendir
    svm.fit(train_features, train_labels)
    val_pred = svm.predict(val_features)
    
    # F1 score döndür
    score = f1_score(val_labels, val_pred, average='weighted')
    return score
```

### 8.3 Optimizasyon Çalıştırma

```python
def optimize_densenet(data_folder, n_trials=20):
    """Run Optuna optimization for DenseNet hyperparameters"""
    
    print(f"Starting DenseNet121 hyperparameter optimization with {n_trials} trials...")
    
    # Study oluştur
    study = optuna.create_study(direction='maximize', study_name='densenet_optimization')
    
    # Optimizasyonu çalıştır
    study.optimize(lambda trial: objective_densenet(trial, data_folder), n_trials=n_trials)
    
    # En iyi deneme
    best_trial = study.best_trial
    
    print(f"Best trial: {best_trial.number}")
    print(f"Best validation accuracy: {best_trial.value:.4f}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Sonuçları kaydet
    trials_df = study.trials_dataframe()
    trials_df.to_csv('densenet_optimization_results.csv', index=False)
    
    # Optimizasyon geçmişini görselleştir
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig('densenet_optimization_history.png', dpi=300)
    plt.show()
    
    # Parametre önemini göster
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig('densenet_param_importances.png', dpi=300)
    plt.show()
    
    return best_trial.params
```

## 9. Ana Uygulama Fonksiyonu

### 9.1 Ana Fonksiyon Yapısı

```python
def run_enhanced_classification_with_optuna(data_folder, run_optimization=True, n_optimization_trials=10):
    print(f"Starting enhanced DenseNet121 classification with Optuna optimization on {data_folder}...")
    
    # 1. BAŞLANGIÇ KONTROLLERI
    if not os.path.exists(data_folder):
        print(f"Error: Data folder '{data_folder}' not found.")
        return None
    
    start_time = time.time()
    
    # 2. HİPERPARAMETRE OPTİMİZASYONU
    if run_optimization:
        best_params = optimize_all_classifiers(data_folder, n_trials=n_optimization_trials)
        # Optimized parametreleri çıkar
        densenet_params = best_params['densenet']
        dense_neurons = densenet_params.get('dense_neurons', 1024)
        dropout_rate = densenet_params.get('dropout_rate', 0.5)
        # ... diğer parametreler
    else:
        # Default parametreler
        dense_neurons = 1024
        dropout_rate = 0.5
# 2. HİPERPARAMETRE OPTİMİZASYONU
    if run_optimization:
        best_params = optimize_all_classifiers(data_folder, n_trials=n_optimization_trials)
        # Optimized parametreleri çıkar
        densenet_params = best_params['densenet']
        dense_neurons = densenet_params.get('dense_neurons', 1024)
        dropout_rate = densenet_params.get('dropout_rate', 0.5)
        optimizer_name = densenet_params.get('optimizer', 'rmsprop')
        learning_rate = densenet_params.get('learning_rate', LEARNING_RATE)
        
        # Data augmentation parametreleri
        aug_rotation = densenet_params.get('aug_rotation', 10)
        aug_width_shift = densenet_params.get('aug_width_shift', 0.1)
        aug_height_shift = densenet_params.get('aug_height_shift', 0.1)
        aug_zoom = densenet_params.get('aug_zoom', 0.1)
    else:
        # Default parametreler kullan
        dense_neurons = 1024
        dropout_rate = 0.5
        optimizer_name = 'rmsprop'
        learning_rate = LEARNING_RATE
        aug_rotation = 10
        aug_width_shift = 0.1
        aug_height_shift = 0.1
        aug_zoom = 0.1
        best_params = {}
    
    # 3. VERİ GENERATORLARİ OLUŞTURMA
    print("Creating data generators with optimized parameters...")
    train_generator, validation_generator, test_generator = create_data_generators(
        data_folder,
        batch_size=BATCH_SIZE,
        aug_rotation=aug_rotation,
        aug_width_shift=aug_width_shift,
        aug_height_shift=aug_height_shift,
        aug_zoom=aug_zoom
    )
    
    # Class bilgilerini al
    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # 4. SONUÇ SAKLAMAK İÇİN YAPILARI HAZIRLA
    all_models_results = {}
    all_models_metrics = []
    
    # En iyi hibrit model takibi
    best_hybrid_model_name = None
    best_hybrid_model_f1 = -1
    best_hybrid_model_predictions = None
    best_hybrid_model_probabilities = None
    best_hybrid_model_true_labels = None
    
    # 5. DENSENET121 MODEL EĞİTİMİ
    print("\n===== Training Optimized DenseNet121 Model =====")
    model = build_densenet_model(
        num_classes=num_classes,
        dense_neurons=dense_neurons,
        dropout_rate=dropout_rate,
        optimizer=optimizer_name,
        learning_rate=learning_rate
    )
    
    # Model eğitimi
    model, history = fast_train(model, train_generator, validation_generator)
    
    # Training history görselleştirme
    plot_training_history(history)
    
    # Model değerlendirme
    densenet_metrics, y_true, y_pred, y_pred_prob = evaluate_densenet(model, test_generator)
    densenet_metrics['model'] = "DenseNet121 (Optimized)"
    
    # Sınıf bazlı metrikler
    densenet_per_class_df = calculate_per_class_metrics(y_true, y_pred, class_names)
    
    # Sonuçları sakla
    all_models_metrics.append(densenet_metrics)
    all_models_results["DenseNet121 (Optimized)"] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_pred_prob
    }
    
    # Confusion matrix çiz
    plot_confusion_matrix(y_true, y_pred, class_names, "DenseNet121 Confusion Matrix")
    
    # DenseNet F1 skorunu sakla
    densenet_f1 = densenet_metrics['f1']
    
    # 6. ÖZELLİK ÇIKARMA İŞLEMİ
    print("\n===== Extracting Features from DenseNet121 =====")
    feature_extractor = create_feature_extractor()
    
    # Generatorları sıfırla
    train_generator.reset()
    validation_generator.reset()
    test_generator.reset()
    
    # Feature extraction
    print("Extracting training features...")
    train_features, train_labels = extract_features(feature_extractor, train_generator)
    print(f"Training features shape: {train_features.shape}")
    
    print("Extracting validation features...")
    val_features, val_labels = extract_features(feature_extractor, validation_generator)
    print(f"Validation features shape: {val_features.shape}")
    
    print("Extracting test features...")
    test_features, test_labels = extract_features(feature_extractor, test_generator)
    print(f"Test features shape: {test_features.shape}")
    
    # 7. SVM ALGORİTMALARI
    print("\n===== Training and Evaluating Optimized SVM =====")
    
    if run_optimization and 'svm' in best_params:
        # Optimized parametrelerle SVM
        svm_params = best_params['svm']
        kernel = svm_params.get('kernel', 'rbf')
        C = svm_params.get('C', 1.0)
        gamma = svm_params.get('gamma', 'scale')
        
        # Polynomial kernel için ek parametreler
        if kernel == 'poly' and 'degree' in svm_params:
            degree = svm_params.get('degree', 3)
            coef0 = svm_params.get('coef0', 0.0)
            # Model eğitimi parametrelerle...
        
        svm_metrics, svm_pred, svm_prob, svm_model = train_and_evaluate_svm(
            train_features, train_labels, test_features, test_labels,
            kernel=kernel, C=C, gamma=gamma
        )
        
        # Sonuçları kaydet
        all_models_metrics.append(svm_metrics)
        all_models_results[f"DenseNet121 + SVM ({kernel})"] = {
            "y_true": test_labels,
            "y_pred": svm_pred,
            "y_prob": svm_prob
        }
        
        # En iyi hibrit modeli kontrol et
        if svm_metrics['f1'] > best_hybrid_model_f1:
            best_hybrid_model_f1 = svm_metrics['f1']
            best_hybrid_model_name = f"DenseNet121 + SVM ({kernel})"
            best_hybrid_model_predictions = svm_pred
            best_hybrid_model_probabilities = svm_prob
            best_hybrid_model_true_labels = test_labels
    else:
        # Optimizasyon yapılmadıysa tüm kernel türlerini dene
        for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
            svm_metrics, svm_pred, svm_prob, svm_model = train_and_evaluate_svm(
                train_features, train_labels, test_features, test_labels, kernel=kernel
            )
            
            all_models_metrics.append(svm_metrics)
            all_models_results[f"DenseNet121 + SVM ({kernel})"] = {
                "y_true": test_labels,
                "y_pred": svm_pred,
                "y_prob": svm_prob
            }
            
            # En iyi hibrit modeli güncelle
            if svm_metrics['f1'] > best_hybrid_model_f1:
                best_hybrid_model_f1 = svm_metrics['f1']
                best_hybrid_model_name = f"DenseNet121 + SVM ({kernel})"
                best_hybrid_model_predictions = svm_pred
                best_hybrid_model_probabilities = svm_prob
                best_hybrid_model_true_labels = test_labels
    
    # 8. BOOSTING ALGORİTMALARI
    print("\n===== Training and Evaluating Optimized Boosting Methods =====")
    
    # 8.1 Gradient Boosting
    if run_optimization and 'gradient_boosting' in best_params:
        gb_params = best_params['gradient_boosting']
        n_estimators = gb_params.get('n_estimators', 100)
        learning_rate = gb_params.get('learning_rate', 0.1)
        max_depth = gb_params.get('max_depth', 3)
        
        gb_metrics, gb_pred, gb_prob, gb_model = train_and_evaluate_gradient_boosting(
            train_features, train_labels, test_features, test_labels,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
    else:
        gb_metrics, gb_pred, gb_prob, gb_model = train_and_evaluate_gradient_boosting(
            train_features, train_labels, test_features, test_labels
        )
    
    # Sonuçları kaydet ve en iyi modeli kontrol et
    all_models_metrics.append(gb_metrics)
    all_models_results["DenseNet121 + Gradient Boosting"] = {
        "y_true": test_labels,
        "y_pred": gb_pred,
        "y_prob": gb_prob
    }
    
    if gb_metrics['f1'] > best_hybrid_model_f1:
        best_hybrid_model_f1 = gb_metrics['f1']
        best_hybrid_model_name = "DenseNet121 + Gradient Boosting"
        best_hybrid_model_predictions = gb_pred
        best_hybrid_model_probabilities = gb_prob
        best_hybrid_model_true_labels = test_labels
    
    # 8.2 XGBoost (Eğer mevcutsa)
    if XGBOOST_AVAILABLE:
        if run_optimization and 'xgboost' in best_params:
            xgb_params = best_params['xgboost']
            n_estimators = xgb_params.get('n_estimators', 100)
            learning_rate = xgb_params.get('learning_rate', 0.1)
            max_depth = xgb_params.get('max_depth', 6)
            
            xgb_metrics, xgb_pred, xgb_prob, xgb_model = train_and_evaluate_xgboost(
                train_features, train_labels, test_features, test_labels,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth
            )
        else:
            xgb_metrics, xgb_pred, xgb_prob, xgb_model = train_and_evaluate_xgboost(
                train_features, train_labels, test_features, test_labels
            )
        
        all_models_metrics.append(xgb_metrics)
        all_models_results["DenseNet121 + XGBoost"] = {
            "y_true": test_labels,
            "y_pred": xgb_pred,
            "y_prob": xgb_prob
        }
        
        if xgb_metrics['f1'] > best_hybrid_model_f1:
            best_hybrid_model_f1 = xgb_metrics['f1']
            best_hybrid_model_name = "DenseNet121 + XGBoost"
            best_hybrid_model_predictions = xgb_pred
            best_hybrid_model_probabilities = xgb_prob
            best_hybrid_model_true_labels = test_labels
    else:
        print("Skipping XGBoost (not installed)")
    
    # 8.3 LightGBM (Eğer mevcutsa)
    if LIGHTGBM_AVAILABLE:
        if run_optimization and 'lightgbm' in best_params:
            lgbm_params = best_params['lightgbm']
            n_estimators = lgbm_params.get('n_estimators', 100)
            learning_rate = lgbm_params.get('learning_rate', 0.1)
            num_leaves = lgbm_params.get('num_leaves', 31)
            
            lgbm_metrics, lgbm_pred, lgbm_prob, lgbm_model = train_and_evaluate_lightgbm(
                train_features, train_labels, test_features, test_labels,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves
            )
        else:
            lgbm_metrics, lgbm_pred, lgbm_prob, lgbm_model = train_and_evaluate_lightgbm(
                train_features, train_labels, test_features, test_labels
            )
        
        all_models_metrics.append(lgbm_metrics)
        all_models_results["DenseNet121 + LightGBM"] = {
            "y_true": test_labels,
            "y_pred": lgbm_pred,
            "y_prob": lgbm_prob
        }
        
        if lgbm_metrics['f1'] > best_hybrid_model_f1:
            best_hybrid_model_f1 = lgbm_metrics['f1']
            best_hybrid_model_name = "DenseNet121 + LightGBM"
            best_hybrid_model_predictions = lgbm_pred
            best_hybrid_model_probabilities = lgbm_prob
            best_hybrid_model_true_labels = test_labels
    else:
        print("Skipping LightGBM (not installed)")
    
    # 8.4 CatBoost (Eğer mevcutsa)
    if CATBOOST_AVAILABLE:
        catboost_metrics, catboost_pred, catboost_prob, catboost_model = train_and_evaluate_catboost(
            train_features, train_labels, test_features, test_labels
        )
        
        all_models_metrics.append(catboost_metrics)
        
        # Sadece geçerli tahminler varsa ROC analizine ekle
        if catboost_prob is not None:
            all_models_results["DenseNet121 + CatBoost"] = {
                "y_true": test_labels,
                "y_pred": catboost_pred,
                "y_prob": catboost_prob
            }
            
            if catboost_metrics['f1'] > best_hybrid_model_f1:
                best_hybrid_model_f1 = catboost_metrics['f1']
                best_hybrid_model_name = "DenseNet121 + CatBoost"
                best_hybrid_model_predictions = catboost_pred
                best_hybrid_model_probabilities = catboost_prob
                best_hybrid_model_true_labels = test_labels
    else:
        print("Skipping CatBoost (not installed or incompatible)")
    
    # 8.5 AdaBoost
    ada_metrics, ada_pred, ada_prob, ada_model = train_and_evaluate_adaboost(
        train_features, train_labels, test_features, test_labels
    )
    
    all_models_metrics.append(ada_metrics)
    all_models_results["DenseNet121 + AdaBoost"] = {
        "y_true": test_labels,
        "y_pred": ada_pred,
        "y_prob": ada_prob
    }
    
    if ada_metrics['f1'] > best_hybrid_model_f1:
        best_hybrid_model_f1 = ada_metrics['f1']
        best_hybrid_model_name = "DenseNet121 + AdaBoost"
        best_hybrid_model_predictions = ada_pred
        best_hybrid_model_probabilities = ada_prob
        best_hybrid_model_true_labels = test_labels
    
    # 9. TÜM MODELLERİ KARŞILAŞTIRMA
    print("\n===== Comparing All Models =====")
    
    if best_hybrid_model_name:
        print(f"Best hybrid model: {best_hybrid_model_name} with F1 score: {best_hybrid_model_f1:.4f}")
        # Baseline ile karşılaştır
        if densenet_f1 > best_hybrid_model_f1:
            print(f"Note: DenseNet121 (Optimized) still outperforms hybrid models with F1 score: {densenet_f1:.4f}")
        else:
            print(f"Hybrid model outperforms DenseNet121 (Optimized) (F1: {densenet_f1:.4f})")
    else:
        print("No viable hybrid models found.")
    
    # 10. TÜM METRİKLERİ DATAFRAME'E ÇEVİR
    all_metrics_df = pd.DataFrame(all_models_metrics)
    
    # En iyi hibrit model için sınıf bazlı metrikler
    best_hybrid_per_class_df = calculate_per_class_metrics(
        best_hybrid_model_true_labels, best_hybrid_model_predictions, class_names
    )
    
    # Metrik tablolarını oluştur ve kaydet
    create_metrics_tables(densenet_per_class_df, best_hybrid_per_class_df)
    
    # Metrikleri 4 ondalık basamağa format la
    for col in ['accuracy', 'precision', 'recall', 'f1', 'f2', 'f0', 'specificity', 'mcc', 'kappa']:
        all_metrics_df[col] = all_metrics_df[col].map(lambda x: f"{x:.4f}")
    
    # Metrikleri göster ve kaydet
    print("\nMetrics for all models:")
    print(all_metrics_df)
    all_metrics_df.to_csv('all_models_metrics.csv', index=False)
    print("Metrics saved to 'all_models_metrics.csv'")
    
    # 11. METRİK GÖRSELLEŞTİRME
    print("\n===== Visualizing Metrics =====")
    
    # String metrikleri float'a çevir
    for col in ['accuracy', 'precision', 'recall', 'f1', 'f2', 'f0', 'specificity', 'mcc', 'kappa']:
        all_metrics_df[col] = all_metrics_df[col].astype(float)
    
    # Görselleştirmeler
    visualize_metrics(all_metrics_df)
    
    # 12. EN İYİ HİBRİT MODEL İÇİN CONFUSION MATRIX
    plot_confusion_matrix(
        best_hybrid_model_true_labels,
        best_hybrid_model_predictions,
        class_names,
        f"{best_hybrid_model_name} Confusion Matrix"
    )
    
    # 13. ROC EĞRİLERİ
    print("\n===== Calculating and Plotting ROC Curves =====")
    roc_df, roc_aucs = calculate_and_plot_roc(all_models_results, class_names, num_classes)
    
    # 14. PRECISION-RECALL EĞRİSİ
    print("\n===== Calculating and Plotting Precision-Recall Curve for Best Hybrid Model =====")
    average_precision = plot_precision_recall_curve(
        best_hybrid_model_true_labels,
        best_hybrid_model_probabilities,
        class_names,
        num_classes,
        best_hybrid_model_name
    )
    
    # 15. ÇALIŞMA SÜRESİ HESAPLAMA
    end_time = time.time()
    execution_time = end_time - start_time
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"\n===== Enhanced Classification Complete =====")
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # 16. SONUÇLARI RETURN ET
    return {
        'all_metrics_df': all_metrics_df,
        'roc_df': roc_df,
        'densenet_per_class_df': densenet_per_class_df,
        'best_hybrid_per_class_df': best_hybrid_per_class_df,
        'best_hybrid_model_name': best_hybrid_model_name,
        'best_hybrid_model_f1': best_hybrid_model_f1,
        'densenet_f1': densenet_f1,
        'best_params': best_params if run_optimization else None,
        'execution_time': execution_time
    }
```

## 10. Kullanım Örnekleri

### 10.1 Temel Kullanım - Optimizasyonsuz

```python
# Hızlı test için optimizayonu kapat
results = run_enhanced_classification_with_optuna(
    "/path/to/dataset",
    run_optimization=False  # Default parametreler kullan
)

print(f"En iyi hibrit model: {results['best_hybrid_model_name']}")
print(f"F1 Score: {results['best_hybrid_model_f1']:.4f}")
```

### 10.2 Optimizasyonlu Kullanım - Az Deneme

```python
# Hızlı optimizasyon
results = run_enhanced_classification_with_optuna(
    "/path/to/dataset",
    run_optimization=True,
    n_optimization_trials=5  # Her algoritma için 5 deneme
)
```

### 10.3 Tam Optimizasyon

```python
# En iyi sonuçlar için
results = run_enhanced_classification_with_optuna(
    "/path/to/dataset",
    run_optimization=True,
    n_optimization_trials=50  # Her algoritma için 50 deneme
)
```

### 10.4 Sonuçları Analiz Etme

```python
# Çalıştırma sonrası analiz
if results:
    # Genel metrikler
    print("\nGenel Performans:")
    print(results['all_metrics_df'])
    
    # ROC AUC karşılaştırması
    print("\nROC AUC Karşılaştırması:")
    print(results['roc_df'])
    
    # DenseNet sınıf bazlı performans
    print("\nDenseNet Sınıf Bazlı Metrikler:")
    print(results['densenet_per_class_df'])
    
    # En iyi hibrit model sınıf bazlı performans
    print("\nEn İyi Hibrit Model Sınıf Bazlı Metrikler:")
    print(results['best_hybrid_per_class_df'])
    
    # Hiperparametre sonuçları (eğer optimizasyon yapıldıysa)
    if results['best_params']:
        print("\nEn İyi Hiperparametreler:")
        for model, params in results['best_params'].items():
            print(f"\n{model}:")
            for param, value in params.items():
                print(f"  {param}: {value}")
```

## 11. İleri Seviye Konfigürasyon

### 11.1 Özel Data Augmentation

```python
# Manuel data generator oluşturma
train_gen, val_gen, test_gen = create_data_generators(
    data_folder,
    batch_size=32,
    val_split=0.2,
    aug_rotation=25,        # Daha agresif rotasyon
    aug_width_shift=0.2,    # Daha fazla horizontal shift
    aug_height_shift=0.2,   # Daha fazla vertical shift
    aug_zoom=0.2,           # Daha fazla zoom
    aug_horizontal_flip=True
)
```

### 11.2 Özel Model Parametreleri

```python
# Özel DenseNet modeli
custom_model = build_densenet_model(
    num_classes=10,
    dense_neurons=2048,     # Daha büyük dense layer
    dropout_rate=0.7,       # Daha yüksek dropout
    optimizer='adam',       # Adam optimizer
    learning_rate=0.0005    # Farklı learning rate
)
```

### 11.3 Bellek Optimizasyonu

```python
# GPU bellek büyümesini etkinleştir
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Küçük batch size kullan
BATCH_SIZE = 16  # Varsayılan 32 yerine

# Düşük resolution kullan
IMAGE_SIZE = (128, 128)  # Varsayılan (150, 150) yerine
```

## 12. Sorun Giderme

### 12.1 Yaygın Hatalar

#### A. Import Hataları
```python
# Kütüphane eksikse bu hata alınabilir
ImportError: No module named 'lightgbm'

# Çözüm:
pip install lightgbm
```

#### B. GPU Memory Hatası
```python
# TensorFlow GPU memory hatası
ResourceExhaustedError: OOM when allocating tensor

# Çözüm 1: Batch size azalt
BATCH_SIZE = 16

# Çözüm 2: GPU memory growth etkinleştir
tf.config.experimental.set_memory_growth(gpu, True)

# Çözüm 3: Mixed precision kullan
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

#### C. Veri Formatı Hataları
```python
# Yanlış klasör yapısı hatası
Found 0 images

# Çözüm: Klasör yapısını kontrol et
# Doğru yapı:
# dataset/
#   ├── train/
#   │   ├── class1/
#   │   ├── class2/
#   └── test/
#       ├── class1/
#       └── class2/
```

### 12.2 Performans Optimizasyonu

#### A. Hızlı Test İçin
```python
# Epoch sayısını azalt
EPOCHS = 10

# Optimizasyon deneme sayısını azalt
n_optimization_trials = 3

# Optimizasyonu kapat
run_optimization = False
```

#### B. En İyi Sonuçlar İçin
```python
# Epoch sayısını artır
EPOCHS = 50

# Optimizasyon deneme sayısını artır
n_optimization_trials = 100

# Tüm opsiyonel kütüphaneleri yükle
pip install lightgbm xgboost catboost
```

### 12.3 Debugging İpuçları

#### A. Model Eğitimi İzleme
```python
# Verbose=1 ile detaylı çıktı
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    verbose=1,  # Progress bar göster
    callbacks=[early_stop]
)
```

#### B. Optuna İzleme
```python
# Optuna dashboard kullan
# Terminal'de çalıştır:
# optuna-dashboard sqlite:///study.db

# Veya manual tracking
def objective_with_logging(trial):
    # Log her trial
    print(f"Trial {trial.number} started")
    print(f"Params: {trial.params}")
    
    result = objective_densenet(trial, data_folder)
    
    print(f"Trial {trial.number} result: {result}")
    return result
```

## 13. Proje Yapısı ve Dosya Organizasyonu

### 13.1 Önerilen Proje Yapısı
```
project_name/
├── main_script.py          # Ana kod dosyası
├── requirements.txt        # Python gereksinimleri
├── README.md              # Bu dokümantasyon
├── config.py              # Konfigürasyon dosyası (opsiyonel)
├── data/                  # Veri klasörü
│   ├── train/
│   └── test/
├── results/               # Sonuç dosyaları
│   ├── metrics/          # CSV metrik dosyaları
│   ├── plots/            # Graf ve görselleştirmeler
│   └── models/           # Kaydedilen modeller
└── logs/                  # Log dosyaları
    └── optimization/      # Optuna log'ları
```

### 13.2 Konfigürasyon Dosyası Örneği
```python
# config.py
class Config:
    # Veri parametreleri
    DATA_FOLDER = "/path/to/dataset"
    IMAGE_SIZE = (150, 150)
    BATCH_SIZE = 32
    VAL_SPLIT = 0.2
    
    # Eğitim parametreleri
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    
    # Optimizasyon parametreleri
    RUN_OPTIMIZATION = True
    N_OPTIMIZATION_TRIALS = 20
    
    # Çıktı klasörleri
    RESULTS_DIR = "./results"
    PLOTS_DIR = "./results/plots"
    METRICS_DIR = "./results/metrics"
```

## 14. Gelecek Geliştirmeler

### 14.1 Potansiyel İyileştirmeler
1. **Ensemble Methods**: Birden fazla modeli kombine etme
2. **Advanced Augmentation**: Cutout, MixUp, CutMix teknikleri
3. **Attention Mechanisms**: Attention tabanlı feature weighting
4. **Multi-Scale Training**: Farklı resolution'larda eğitim
5. **Semi-Supervised Learning**: Unlabeled data kullanımı

### 14.2 Model Arşitektura Alternatifleri
```python
# EfficientNet kullanımı
from tensorflow.keras.applications import EfficientNetB0

# ResNet kullanımı
from tensorflow.keras.applications import ResNet50V2

# Vision Transformer
# from transformers import ViTFeatureExtractor, ViTForImageClassification
```

## 15. Katkıda Bulunma

### 15.1 Development Setup
```bash
# Development ortamı kurma
git clone <repository>
cd densenet121-classification
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements-dev.txt
```

### 15.2 Code Style
```python
# Black formatter kullan
pip install black
black main_script.py

# Docstring standartları
def function_name(param1, param2):
    """
    Brief description.
    
    Args:
        param1 (type): Description
        param2 (type): Description
    
    Returns:
        type: Description
    """
    pass
```

## 16. Lisans ve Referanslar

### 16.1 Açık Kaynak Kütüphaneler
- **TensorFlow/Keras**: Apache License 2.0
- **Scikit-learn**: BSD License
- **Optuna**: MIT License
- **Pandas**: BSD License
- **NumPy**: BSD License

### 16.2 Bilimsel Referanslar
1. DenseNet: Huang, G., et al. "Densely connected convolutional networks." (2017)
2. Optuna: Akiba, T., et al. "Optuna: A next-generation hyperparameter optimization framework." (2019)
3. Transfer Learning: Pan, S. J., & Yang, Q. "A survey on transfer learning." (2010)
