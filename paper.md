---
title: 'World Happiness: a tested analysis toolkit and dashboards for the World Happiness Report'
tags:
  - Python
  - data science
  - reproducibility
  - causal inference
authors:
  - name: Abdullahi Adinoyi Ibrahim
    affiliation: 1
affiliations:
  - name: TU Dresden, Germany
    index: 1
date: 30 September 2026
bibliography: paper.bib
---

# Summary

`world-happiness` turns the World Happiness Report *Data for Figure 2.1* into a validated,
ISO3-keyed country-year panel. On that panel it provides descriptive analysis, an XGBoost
model with grouped and temporal validation and TreeSHAP explanations, and DoWhy back-door
estimates. A Streamlit explorer and a Dash results dashboard present the outputs, and the
whole pipeline runs in one command or one `docker compose up`.

# Statement of need

TODO: position against existing WHR analyses and state the intended audience (teaching,
reproducible baselines). See MODEL_CARD.md for the limits of the factor data.

# Acknowledgements

Data: World Happiness Report / Gallup World Poll.
