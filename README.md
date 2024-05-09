# MachineLearningDSJobs

■ jaki problem będzie rozwiązywany z użyciem uczenia maszynowego?
  Prognozowanie wynagrodzeń w dziedzinie Data Science oraz analiza wpływu czynników związanych z doświadczeniem zawodowym, typem zatrudnienia i stopniem pracy zdalnej na wysokość wynagrodzenia. Oraz wpływ czynników na koeficjent pracy zdalnej.
■ na podstawie jakich danych historycznych?
  Na podstawie danych z Kaggle.com (https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries)
  Struktura danych:
  work_year: Rok, w którym wypłacono wynagrodzenie.
  experience_level: Poziom doświadczenia w pracy w danym roku, z kategorią:
  EN: Poziom początkujący / Junior
  MI: Poziom średni / Pośredni
  SE: Poziom starszy / Ekspert
  EX: Poziom wykonawczy / Dyrektor
  employment_type: Typ zatrudnienia dla danej roli, z kategoriami:
  PT: Praca w niepełnym wymiarze godzin
  FT: Praca w pełnym wymiarze godzin
  CT: Praca na umowę
  FL: Praca na zlecenie
  job_title: Stanowisko pracy w danym roku.
  salary: Całkowita kwota brutto wypłaconego wynagrodzenia.
  salary_currency: Waluta, w której wypłacono wynagrodzenie, w formie kodu waluty ISO 4217.
  salary_in_usd: Kwota wynagrodzenia przeliczona na USD, obliczona na podstawie kursu wymiany walut podzielonego przez średni kurs USD dla danego roku.
  employee_residence: Główny kraj zamieszkania pracownika w danym roku, w formie kodu kraju ISO 3166.
  remote_ratio: Ogólny odsetek pracy wykonywanej zdalnie, z możliwymi wartościami:
  0: Brak pracy zdalnej (mniej niż 20%)
  50: Częściowo zdalna
  100: W pełni zdalna (powyżej 80%)
  company_location: Kraj głównego biura pracodawcy lub oddziału kontraktowego, w formie kodu kraju ISO 3166.
  company_size: Średnia liczba osób pracujących dla firmy w danym roku, z kategoriami:
  S: mniej niż 50 pracowników (mała firma)
  M: od 50 do 250 pracowników (średnia firma)
  L: więcej niż 250 pracowników (duża firma)
