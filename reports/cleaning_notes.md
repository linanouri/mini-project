# Cleaning Decisions

1. Missing target values (`primary_type`) were dropped because supervised classification requires known labels.
2. Numeric features were median-imputed to reduce sensitivity to skew and outliers.
3. Exact duplicate Pokemon IDs were removed defensively.
4. Outliers were reviewed but retained because extreme stats can be valid Pokemon characteristics.
5. Very rare classes (<3 rows) were mapped to `other` to stabilize stratified splitting and macro metrics.

## Dataset Summary

- Raw rows: 300
- Raw columns: 14
- Clean rows: 300
- Clean columns: 14
