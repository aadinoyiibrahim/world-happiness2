# Model card

## Data

- **Source:** WHR 2026 *Data for Figure 2.1*, 2,116 country-years, 167 countries, 2011–2025
  (no 2013). `year` is the last year of each 3-year average.
- **Factor decomposition:** available from 2019 only (1,013 rows with all six factors; 6
  partial rows are left out of modelling). The six "explained by"
  columns plus `dystopia_residual` add up to the score (max rounding error 0.003).
- **Cleaning:** names are mapped to ISO3 codes (`Swaziland` → `Eswatini`; North Cyprus
  and Somaliland get their own codes, `CYN` and `SOL`). Countries are assigned to the ten
  WHR regions, and no row is left unmapped. Missing factors stay `NaN`; they are never
  filled with 0. A pandera schema checks value ranges, that each score lies inside its CI,
  and that each (iso3, year) appears only once.

## Predictive model

- **Target:** `happiness_score`. **Features:** the six factor contributions.
  `dystopia_residual` and anything derived from the target are excluded.
- **Models:** XGBoost (depth 3, 400 trees), OLS, and an *additive identity* baseline
  (the sum of the factors plus a constant, with nothing learned).
- **Validation:** *grouped* is 5-fold cross-validation grouped by country, so it tests on
  unseen countries. *Temporal* trains on 2019–2024 and tests on 2025.
- **Results** (`outputs/metrics.csv`, held-out R²):

  | scheme | additive | linear | XGBoost |
  |---|---|---|---|
  | grouped | 0.65 | 0.67 | 0.69 |
  | temporal | 0.45 | 0.52 | 0.76 |

## Caveats

1. **The features are not raw measurements.** Each contribution is WHR's pooled regression
   coefficient multiplied by the gap to the "Dystopia" benchmark. Across countries, a
   linear model on these features is little more than WHR's own model; XGBoost only adds
   non-linear terms. WHR's free data no longer includes the raw indicators, so moving to
   them needs Gallup access or the WHR 2024 *Data for Table 2.1* file (2005–2023).
2. **The temporal score is optimistic.** Consecutive 3-year averages share two of their
   three survey years, so the 2025 test set overlaps heavily with the training years.
   Treat the grouped score as the honest number.
3. **Coefficients vary across editions.** Each WHR edition re-estimates its coefficients,
   so the same contribution can mean slightly different things in different years.
4. **Causal estimates depend on an assumed DAG.** The DAG (`causal.py`) has income and
   health driving the other factors. For each factor we report two effects, both with 95%
   CIs clustered by country:
   - the **total effect**: DoWhy back-door, which includes paths through other factors;
   - the **direct effect**: all other factors held fixed.

   If the WHR decomposition carried over exactly to this pooled panel, every *direct* effect
   would be 1. Instead GDP (0.64) and freedom (0.48) fall below 1, and health (1.34) sits
   at the edge above it. So the decomposition does not transfer cleanly: coefficients vary
   between editions, and the residual is correlated with the factors. Treat these numbers
   as diagnostics of the decomposition, not as real-world causal effects.
5. **Regional means mix in composition changes.** The set of surveyed countries differs
   from year to year.
