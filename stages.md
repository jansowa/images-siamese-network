# Etapy projektu
1. Zrozumienie podstaw
   1. Zrozumienie podstawowych pojęć sieci neuronowych takich jak neuron, funkcja straty, 
   2. Wykład o podstawach architektury Siamese Network
2. Uruchomienie dowolnego modelu implementującego architekturę Siamese Network, który jest trenowany na zdjęciach owoców:
   1. Przygotowanie środowiska lokalnego (preferowane - Conda) lub w serwisie Kaggle (karta 2x Tesla T4 / 1x Tesla P100 / TPU VM v3-8)
   2. Uruchomenie gotowego skryptu, np. [contrastive loss z dokumentacji keras](https://keras.io/examples/vision/siamese_contrastive/) lub [triplet loss z dokumentacji keras](https://keras.io/examples/vision/siamese_network/)
   3. Proste zbiory: [pierwszy](https://www.kaggle.com/datasets/moltean/fruits), [drugi](https://www.kaggle.com/datasets/chrisfilo/fruit-recognition)
3. Przygotowanie narzędzi pobocznych:
   1. Wstawienie zdjęć (wzorców) do bazy (foldera), klasyfikowanie na ich podstawie całego zbioru testowego. Wyliczanie dobranych metryk (np. średnie miejsce, top1 accuracy, top3 accuracy)
   2. Implementacja funkcji do zapisu i odczytu modeli (np. metodami [save_weights](https://keras.io/api/models/model_saving_apis/weights_saving_and_loading/#save_weights-method), [load_weights](https://keras.io/api/models/model_saving_apis/weights_saving_and_loading/#load_weights-method))
   3. Generowanie z folderów ze zdjęciami obiektów gotowych do trenowania
   4. Wyświetlanie błędnie sklasyfikowanych zdjęć
4. Implementacja kodu umożliwiającego fine-tuning bazowej sieci. Opcjonalnie - zapis 'bottleneck features' [do pliku](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).
5. Eksperymenty mające na celu poprawę jakości modelu
6. Przygotowanie podsumowania modelu z osiągniętymi wynikami