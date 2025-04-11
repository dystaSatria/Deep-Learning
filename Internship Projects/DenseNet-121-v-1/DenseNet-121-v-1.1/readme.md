# V-1.1

- F(0,5)

# V-1.2

- F(0)


- https://t.me/dystaSatriaFiles/22

## F0 and F0.5 Kıyaslaması

F0 ve F0.5, sınıflandırma modellerinin performansını değerlendirmek için kullanılan F-beta skorlarının farklı versiyonlarıdır. İşte aralarındaki temel farklar:
F0 (Beta = 0)

F0, beta değeri 0 olan F-beta skorudur
Matematiksel olarak, F0 tam olarak Precision (Kesinlik) değerine eşittir
Recall (Duyarlılık/Geri Çağırma) değerini hiç dikkate almaz
Formül: F0 = (1 + 0²) × (precision × recall) / (0² × precision + recall) = precision
Yalnızca yanlış pozitifleri (false positives) önemser, yanlış negatifleri (false negatives) hiç önemsemez
Ne zaman kullanılır: Yanlış pozitiflerin maliyetinin çok yüksek olduğu ve yanlış negatiflerin önemsiz olduğu durumlarda

F0.5 (Beta = 0.5)

F0.5, beta değeri 0.5 olan F-beta skorudur
Precision'a Recall'dan daha fazla ağırlık verir (2 kat daha fazla)
Formül: F0.5 = (1 + 0.5²) × (precision × recall) / (0.5² × precision + recall)
Hem yanlış pozitifleri hem de yanlış negatifleri dikkate alır, ancak yanlış pozitiflere daha fazla ağırlık verir
Ne zaman kullanılır: Yanlış pozitiflerin maliyetinin yanlış negatiflerden daha yüksek olduğu, ancak yanlış negatiflerin de tamamen göz ardı edilemeyeceği durumlarda
