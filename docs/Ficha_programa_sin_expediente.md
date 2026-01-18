# Ficha: app_sin_expediente.py

**Versión**: 1.0.0 | **Fecha**: 2025-01-18

## Descripción

Dashboard interactiu per a l'anàlisi comparatiu de la contractació sense expedient entre els exercicis 2024 i 2025. Permet identificar increments significatius, nous proveïdors i analitzar en detall sectors crítics com viatges i publicacions científiques.

## Entrada

- `datos/SIN EXPEDIENTE.xlsx`:
  - Fulla `SIN EXPTE 2024`: Registres de contractació sense expedient 2024
  - Fulla `SIN EXPTE 2025`: Registres de contractació sense expedient 2025
  - Columnes principals: `Nombre Complet`, `N Factura`, `Desc Gasto`, `Base imp`

## Processament

- Classificació automàtica de gastos en categories (publicacions, viatges, membresies, etc.)
- Càlcul de mètriques comparatives 2024 vs 2025
- Identificació de nous proveïdors
- Anàlisi específic de sectors crítics

## Mòduls

| Mòdul | Descripció |
|-------|------------|
| ℹ️ Presentació i Dades | Visió general i fonts de dades |
| 📊 Resum Executiu | Mètriques principals i indicadors clau |
| 📈 Comparativa per Categories | Evolució per tipus de gasto |
| ✈️ Anàlisi de Viatges | Detall de proveïdors de viatges i transport |
| 📚 Anàlisi de Publicacions | Detall d'editorials i publicacions científiques |
| 🏢 Top Proveïdors | Rànquing i increments per proveïdor |
| 📋 Detall de Registres | Taula filtrable amb exportació a Excel |

## Eixides

- Visualitzacions interactives (Plotly)
- Exportació a Excel (`salidas/*.xlsx`)
- Caché Parquet per a càrrega ràpida (`parquet/*.parquet`)

## Dependències

```
pandas
numpy
plotly
streamlit
openpyxl
pyarrow
```

## Ús

```bash
# Desde WSL
cd ~/claude-test-project/SCAG
source ../venv_python/bin/activate
streamlit run app_sin_expediente.py

# Desde Windows
Ejecutar: Iniciar_Dashboard_Sin_Expediente.bat
```

## Resultats Clau Identificats

| Mètrica | 2024 | 2025 | Variació |
|---------|------|------|----------|
| Registres | 1.361 | 2.385 | +75% |
| Import total | 662.574 € | 1.213.342 € | +83% |
| Proveïdors | 525 | 895 | +70% |
| Nous proveïdors | - | 631 | 54% del total |

## Sectors Crítics

1. **Viatges**: Noves agències (Rosselli, Mago Tours, Mediterráneo Holidays) ~100K€
2. **Publicacions**: ACS Publications (70K€), increment Elsevier/Springer
3. **Subministraments**: FACSA (+24K€)
