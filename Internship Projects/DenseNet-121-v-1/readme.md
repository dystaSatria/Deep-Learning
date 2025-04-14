# DenseNet 121 Version of 1 

Problem :
- To slow to train this data which wasting so many times

Çözüm :

- keras.applications.DenseNet121'den önceden eğitilmiş bir model kullanıyor olabilir.

- DenseNet üzerine ağır özelleştirilmiş katmanlar eklememiş olanı kullanırım.
-  Eğitim Stratejisi : ince ayar (fine-tuning) yerine feature extraction kullanırım
- ön işleme kısmı daha basitleştirdim
- early stopping veya learning rate scheduler gibi geri çağırmalar (callback) olabilir. (süpheliyim)
- daha büyük batch size. GPu hızlı olduğundan dolayı hızlı olur
-  early stopping veya learning rate scheduler (callback) . O yüzden daha verimli ve daha erken duruyor
