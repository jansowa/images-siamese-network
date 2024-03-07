# Materiały edukacyjne, porady i przydatne funkcje
Nie traktujcie wszystkich tych materiałów jako obowiązkowe. Warto przeczytać sam tekst bezpośrednio z tego pliku, ale odnośniki zewnętrzne traktujcie jako opcjonalne - przydatne w szczególności, kiedy macie problem z konkretnym etapem projektu.

## Etap 1 - zrozumienie podstaw architektury Siamese Network
1. [Skrypt](./lecture-script.md) mojego wykładu.
2. Krótkie [wyjaśnienie](https://www.youtube.com/watch?v=6jfw8MuKwpI) architektury przez Andrew NG.

## Etap 2 - budowa najprostszej, działającej wersji modelu
### Informacje dotyczące przygotowywania danych
Szczególnie przy dużych zbiorach danych musicie uważać, żeby zaciągać dane w sposób leniwy. Zazwyczaj kiepskim pomysłem będzie zaciągnięcie kilku gigabajtów do pamięci RAM, a następnie odpalenie na tej samej maszynie dużego modelu. Do tego celu będziemy używali klasy ```tf.data.Dataset```. Nadal - musicie zwracać uwagę na to, jak tworzycie taki dataset.
Kilka sposobów na realizację tego zadania:
1. Użycie metody ```image_dataset_from_directory```.
2. Można wczytać do generatora Python same ścieżki plików, a dopiero w pętli przerabiać je na obrazy i wysyłać je za pomocą "yield". Do zrobienia datasetu z takiego generatora użyjecie metody ```from_generator```.
3. Tworzycie listę ścieżek (być może pary, trójki - zależnie od konkretnej implementacji) plików z etykietami, przerabiacie na dataset za pomocą ```from_tensor_slices```, a później tworzycie pożądany zapis zdjęć za pomocą ```dataset.map```.

Jakie sposoby mogą powodować problemy pomimo użycia klasy ```Dataset```? Przykłady:
1. Wczytanie wszystkich plików to tablicy numpy, a dopiero później stworzenie z tego generatora. 
2. Wczytanie wszystkich plików do tablicy numpy, a następnie utworzenie z niego instancji klasy ```Dataset```. 
W obydwu przypadkach na samym początku zaciągamy wszystkie zdjęcia do pamięci RAM. 

Przydatne funkcje:
1. ```tf.data.Dataset.zip``` - łączenie kilku datasetów
2. ```dataset.batch(batch_size)``` - umożliwia ustalenie, po ile elementów ma być w jednej partii ('batch'). Bez tego model może informować, że brakuje mu jednego wymiaru w danych (uwaga - ```image_dataset_from_directory``` już dzieli dataset na batch-e, można tym sterować poprzez odpowiedni parametr tej metody) 
3. ```tf.keras.utils.image_dataset_from_directory``` - tworzenie datasetu, od razu z etykietami
4. ```tf.data.Dataset.from_generator``` - umożliwia implementację standardowego "Pythonowego" generatora, a następnie przerobienie go na dataset
5. ```tf.data.Dataset.from_tensor_slices``` - tworzy dataset z tensorów, ale też np. ze standardowych list

Przydatne materiały:
1. [Artykuł z PyImageSearch](https://pyimagesearch.com/2023/02/13/building-a-dataset-for-triplet-loss-with-keras-and-tensorflow/) - sposób na utworzenie losowego generatora do triplet loss. Zwróć uwagę na to, że:
-to generator nieskończony, ma to swoje konsekwencje w aspekcie uruchamiania modelu (łatwo przerobić jednak na zwracanie konkretnej liczby elementów) 
-za każdym razem zwróci inne elementy
-łatwo przerobić go na generator zwracający parę zdjęć i etykietę (1 dla tych samych klas, 0 dla różnych) - można w ten sposób utworzyć dataset dla modelu z 'contrastive loss'
2. [Dyskusja stackoverflow](https://stackoverflow.com/questions/41064802/l2-normalised-output-with-keras) ze sposobem na zastosowanie normalizacji L2 w ostatniej warstwie.
3. [Contrastive loss z dokumentacji keras](https://keras.io/examples/vision/siamese_contrastive/) - prosty model pracujący na popularnym zbiorze MNIST
4. [Triplet loss z dokumentacji keras](https://keras.io/examples/vision/siamese_network/) - model oparty na sieci ResNet50 (metoda transfer learning), trudniejszy zbiór danych z większymi zdjęciami

Przeglądających zbiory danych zawsze pamiętajcie, żeby trochę je ręcznie przejrzeć. Możecie w nich natrafić na różne nieprawidłowości. Przy okazji wiedza, którą uzyskacie w ten sposób, może Wam ułatwić implementację skuteczniejszego modelu.
Zbiory danych:
1. [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) - bardzo prosty zbiór. Może być dobry na sam początek, żeby ocenić czy nasz model nie zawiera poważnych błędów, czy w ogóle wylicza mniej-więcej to, co powinien.
2. [Fruit recognition](https://www.kaggle.com/datasets/chrisfilo/fruit-recognition) - nasz główny zbiór. Ponad 40 tysięcy zdjęć owoców na wagach/tacach.