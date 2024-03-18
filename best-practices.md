# Dobre praktyki

## Spis treści
0. [Wprowadzenie](#wprowadzenie)
1. [Nazwa gałęzi](#nazwa-gałęzi)
2. [Język komentarzy, nazw zmiennych](#język-komentarzy-nazw-zmiennych)
3. [Jupyter Notebook - nagłówki, wypisywanie wyjścia](#jupyter-notebook---nagłówki-wypisywanie-wyjścia)
  <br/>3.1 [Nagłówki](#nagłówki)
  <br/>3.2 [Wypisywanie wyjścia](#wypisywanie-wyjścia)
4. [Dzielenie kodu pomiędzy Jupyter Notebook, a pliki .py](#dzielenie-kodu-pomiędzy-jupyter-notebook-a-pliki-py)

### Wprowadzenie
1. Elastyczność - nie traktujcie poniższych reguł jako sztywne zasady, a raczej sugestie. Zazwyczaj będą dobrym punktem wyjścia, jednak pamiętajcie, że przy uzasadnionych sytuacjach nie musicie koniecznie trzymać się ich w takiej wersji.
2. Zasadność - na pierwszy rzut oka część z tych reguł może sprawiać wrażenie bezsensownych, lub niepotrzebnie czasochłonnych. Jako cel projektu założyłem nie tylko implementację dobrego modelu, ale też nauczenie Was pewnych zasad istotnych przy projektach komercyjnych. Świat akademicki i biznesowy w pewnych aspektach różnią się między sobą. Przeglądając te reguły, miejcie na uwadze, że chcę Was przygotować do pracy w międzynarodowych projektach, przy których pracują duże zespoły, a czasami trzeba je utrzymywać dekadami. Mając na uwadze, że ktoś innej narodowości i z innymi kompetencjami może poprawiać nasz kod za 10 lat (czasem pod wpływem stresu i w pośpiechu, bo coś "płonie" na produkcji), musimy nauczyć się pewnego stylu programowania.

### Nazwa gałęzi

Powinna odzwierciedlać to, jakie zadanie obecnie w niej realizujecie. Jeśli chcecie mieć kilka równoległych gałęzi z tym samym zagadnieniem, dopuściłbym ewentualnie sufiksy, np. "ver2", " ver3", lub opisujący konkretny wariant zadania - np. użytą bibliotekę. Przy bardziej rozbudowanych projektach (np. z Jirą) często wpisuje się też typ (np. feature, bug) i numer zagadnienia. Zazwyczaj używa się w miarę krótkich nazw z małymi literami, myślnikami i cyframi.

Przykłady sensownych nazw:
- ```first-triplet-loss-model```
- ```feature/first-contrastive-loss-model```

Przykłady problematycznych nazw:
- ```Triplet Loss``` (zastosowanie dużych liter i spacji [może spowodować problemy niektórych skryptów])
- ```marek-branch``` (enigmatyczna, nie wiadomo, w jakim celu została utworzona)
- ```bugfix``` (już minimalnie bardziej, ale nadal mało informatywna)
- ```third-try-of-implementing-simple-siamese-network-with-triplet-loss-version-from-new-google-paper``` (zbyt długa nazwa)

### Język komentarzy, nazw zmiennych
O ile prowadzący nie wymaga tego od Was - nie używajcie języka polskiego ani w komentarzach, ani bezpośrednio w kodzie. Wiąże się to z kilkoma konsekwencjami:
- problemy w pracy z osobami, które nie znają języka polskiego
- mieszanie ze sobą dwóch języków, które prowadzą do tak dziwnych połączeń, jak słynne [isPies](https://wykop.pl/wpis/30442309/kod-pkp-jest-zlotem-function-czywybranopsa-var-isp) - wpływa negatywnie na czytelność nawet dla polskojęzycznych osób
- brak możliwości zaprezentowania kodu na takich serwisach jak Kaggle

Dopuszczam nieliczne wyjątki od tej reguły. Przede wszystkim:
- nazwy własne - jeśli samodzielnie przetłumaczycie np. nazwę usługi, programu, może to utrudnić zrozumienie innym osobom
- wyrażenia, które nie mają dobrych odpowiedników w języku angielskim, a jednocześnie są powiązane typowo z polskim prawem, kulturą i innymi lokalnymi kontekstami. Mógłbym mieć poważne problemy ze znalezieniem kodu odpowiedzialnego za szkolenia BHP, gdyby ktoś zapisał ```OccupationalHygieneAndSafetyTrainingModule``` ;)

### Jupyter Notebook - nagłówki, wypisywanie wyjścia
#### Nagłówki
Warto ich użyć do podziału całego notatnika na etapy (np. ładowanie danych, wizualizacja danych, tworzenie modelu, trenowanie modelu, testowanie modelu). Ułatwi to przeglądanie kodu, zwłaszcza osobom, które go nie pisały. Aby utworzyć nagłówki, należy użyć komórki z językiem "Markdown". W tym języku nagłówki tworzy się za pomocą znaków "#". Im więcej znaków "#", tym ***mniejszy*** będzie nagłówek. Pamiętajcie o znaku spacji pomiędzy znakami "#" a tekstem. W komórkach tekstowych możecie dodawać także inne komentarze.

#### Wypisywanie wyjścia
Co jakiś czas warto wypisać aktualny stan pewnych wartości, zmiennych dla:
- upewnienia się, że faktycznie zawierają to, czego się spodziewamy
- łatwiejszego zrozumienia kodu (ponownie - głównie dla innych osób, ale też dla nas - w szczególności, gdy wracamy do projektu po przerwie)

Możemy np. wyświetlać rozmiar wczytanych danych (w numpy służy do tego funkcja  ```.shape```), kilka losowych zdjęć, architekturę sieci (metoda ```model.summary()```), nie bójcie się też wrzucać do repozytorium komórki z komunikatami błędów. Dzięki temu drugiej osobie łatwiej będzie pomóc.

### Dzielenie kodu pomiędzy Jupyter Notebook, a pliki .py.
Jupyter Notebook ma swoje zalety (przechowywanie zmiennych w pamięci, możliwość wielokrotnego uruchamiania dalszych linijek już po np. wczytaniu danych i kompilacji modelu), ale też wady. Najpopularniejsze IDE nie oferują wsparcia dla formatu  .ipynb. W praktyce zazwyczaj najbardziej opłacało się będzie korzystać równolegle z notatników i standardowych plików .py. Ciężko podać tutaj jakieś sztywne reguły, jednak notatniki dobrze sprawdzają się w prototypowaniu i eksperymentach, a pliki .py do refaktoryzacji, implementacji bardziej złożonych funkcji itp. Sam często najpierw piszę nieco chaotyczny kod w notatniku, następnie przenoszę go do IDE, a tam poprawiam - wydzielam mniejsze funkcje, ustalam lepsze nazwy zmiennych, poprawiam czytelność implementacji itp. Konkretny podział zależy nawet od preferencji programisty.