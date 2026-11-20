---
name: data-preprocessing
description: Diagnose, clean, integrate, transform, reduce, anonymize, and validate datasets for analysis or machine learning. Use for CSV, Excel, JSON, database exports, time-series, and video-derived skeleton/pose data.
---

# Data Preprocessing Skill

## Purpose
Convert raw datasets into analysis- or model-ready data while preserving meaning, traceability, and reproducibility. Choose methods based on the data and task; do not apply every technique by default.

## Principles
- Inspect before changing; preserve an untouched copy of source data.
- Explicitly document rules, parameters, affected records, and reasons.
- Distinguish verified errors from unusual but valid observations.
- Validate after each major operation; retain lineage for material changes.
- For predictive workflows, fit learned preprocessing on training data only and apply fitted parameters to validation/test data.
- Avoid unnecessary information loss and never silently coerce or discard data.

## Scope
Tabular and structured data (CSV, TSV, Excel, JSON, database exports), numeric/categorical/text/date fields, time series/events, and video-derived pose/skeleton/keypoint datasets (frame/timestamp, bbox, keypoints, confidence, image size).

## Data-quality dimensions
Assess:
- Completeness: missing values, rows, frames, required fields.
- Accuracy: invalid values, measurement/extraction errors.
- Consistency: units, types, definitions, formats, constraints.
- Timeliness: freshness, timestamp order, delays.
Also assess uniqueness, validity, integrity, and representativeness as relevant.

## Workflow
1. Clarify goal, target (if any), intended analysis, output format, and constraints.
2. Preserve raw input; load data and record provenance.
3. Profile schema, shape, types, nulls/sentinels, duplicates, ranges, distributions, units, keys, timestamps, and sample records.
4. Categorize issues: missingness, invalid values/outliers, noise, duplicates/redundancy, type/format conflicts, source-integration conflicts.
5. Propose explicit preprocessing rules and parameters before destructive operations.
6. Clean; integrate sources if required; transform; reduce only when justified; protect sensitive data as needed.
7. Validate outputs and export processed data plus report, config, and logs.

## Profiling
Check row/column counts and duplicate column names; data types; missing counts and percentages; empty strings and sentinel values; exact duplicates and duplicate keys; numeric summaries and categorical frequencies; date/time parsing, timezone, order, gaps; units, value domains, relationships, and key constraints. Do not assume digit-like identifiers are numeric (leading zeros may matter). Verify sentinel values (e.g., -1, 999, NA) before treating them as missing.

Example:
\`\`\`python
import pandas as pd
df = pd.read_csv("input.csv")
print(df.shape, df.dtypes, sep="\\n")
print(df.head())
print(df.isna().sum())
print(df.duplicated().sum())
print(df.describe(include="all"))
\`\`\`

## Cleaning
### Missing values
Determine what blank/null/sentinel values mean. Methods: mean (approximately symmetric numeric, low outlier sensitivity), median (skew/outliers), mode (categorical/discrete), justified group-wise imputation, relationship-based inference, forward/back fill only on valid ordered data within the same entity, row deletion when loss is small and unbiased, column deletion when unneeded/unusable. Document assumptions and compare missingness before/after. Fit model-input imputation on training data only.

### Outliers and invalid values
Use domain limits, logic rules, IQR, robust statistics, z-scores, temporal detection, or multivariate methods. Separate impossible/erroneous values from valid extremes. Do not automatically delete: correct known errors, otherwise retain, flag, investigate, or use robust methods. If clipping/winsorizing, record thresholds and impact.

### Noise
Use binning, smoothing, regression, clustering, or signal-specific filtering only when appropriate. Preserve originals; verify that filtering does not remove real events or dynamics.

### Duplicates/redundancy
Check exact duplicates, duplicate keys, repeated observations, redundant columns, and conditional duplicates. Define keys and a retention rule (e.g., latest valid timestamp, source priority, quality). Similar values may represent valid repeated measurements.

### Type/format consistency
After confirming meaning, standardize numeric/categorical types, strings, dates/timezones, units, casing, whitespace, and allowed categories. Use explicit unit conversions and cross-field validation. Log conversion failures rather than silently turning all failures into null.

Common cleaning sequence: normalize missing-value representations and types; correct invalid records; resolve duplicates; impute; handle remaining anomalies; revalidate. Revisit sequence if needed.

## Integration
Before joins, verify key uniqueness and join cardinality (one-to-one, one-to-many, many-to-many). Align names, meanings, types, units, category definitions, timezones, and granularity. Select merge/concat/union based on source relationship. Resolve conflicts with documented source priority, timestamp, quality, or domain rules. Validate row counts, unmatched keys, row multiplication, duplicates, and consistency after integration.

## Transformation
- Generalize details into broader categories where appropriate (e.g., age bands).
- Construct features with documented formulas; guard against divide-by-zero and invalid inputs.
- Encode/map categories; retain mappings and define behavior for unseen values.
- Treat noise using justified methods (binning, smoothing, regression, clustering).
- Scale only when required by the algorithm or analysis.

### Scaling methods
- Min-Max: \`x_scaled = (x - min) / (max - min)\` using training-set extrema. Handle constant features; future values can exceed [0,1] unless clipping is specifically required.
- Z-score: \`z = (x - mean) / std\` using training-set statistics. Handle zero variance.
- Decimal scaling: \`x_scaled = x / 10^j\`; document j.
- Alternatives: robust (median/IQR), MaxAbs, L1/L2, log, quantile. Check domain restrictions and distortions.

## Reduction
Only when justified: remove irrelevant/redundant/constant fields; sampling/aggregation/compression; PCA or other dimensionality reduction; temporal resampling/frame sampling. Document retained features, method, parameters, and information loss. Preserve rare classes, critical events, and needed variation. Fit learned reduction on training data only.

## Privacy
Identify direct identifiers and sensitive attributes. Use masking, removal, pseudonymous stable IDs (protect linkage keys), generalization, or suppression as appropriate. Unsalted hashes of low-entropy identifiers may be guessable. Removing names alone does not guarantee anonymity: consider quasi-identifiers, small groups, linkage risk, and combinations. Minimize sensitive values in logs/reports.

## Excel and pandas string examples
- Excel: \`RIGHT(A2,n)\` extracts rightmost characters; \`LEN(A2)\` counts characters; \`REPLACE\` is position-based and \`SUBSTITUTE\` replaces matching text. Confirm intended behavior.
- pandas:
\`\`\`python
df["column"] = (df["column"].astype("string")
                 .str.replace("unwanted_text", "", regex=False)
                 .str.strip())
\`\`\`
Use narrow, validated replacement rules; avoid deleting valid characters broadly.

## Video pose/skeleton preprocessing
Inspect clip/subject IDs, frame/timestamp/fps, image size, bounding boxes, keypoint schema/order, coordinate units/system, confidence/visibility, missing/occluded joints, frame gaps/order, and subject alignment. Check invalid ranges, implausible jumps, left/right swaps, confidence thresholds, and schema mismatches. Treat low-confidence joints as uncertain/missing; do not interpolate across long gaps or subject changes. Preserve raw coordinates and bbox.

For bbox width \`w\`, height \`h\`, center \`(X_c,Y_c)\`, and joint \`(X_i,Y_i)\`:
- Center-relative: \`X'_i = X_i - X_c\`, \`Y'_i = Y_i - Y_c\`.
- Bbox-relative: \`x_i = (X_i - X_c)/w\`, \`y_i = (Y_i - Y_c)/h\`, so \`P_i=(x_i,y_i)\`.
Require positive, nonzero w/h. Keep joint order and coordinate convention consistent. This normalizes translation and bbox scale, but does not eliminate camera angle, perspective, occlusion, or pose-estimation error. If smoothing, ensure timing and salient motion are preserved.

## ML leakage prevention
Split data before fitting learned preprocessing. Fit imputers, scalers, encoders, feature selection, PCA, etc. on training only; transform validation/test using fitted parameters. Use time-aware or group-aware splits to prevent future leakage or overlap of subjects/entities. Do not include target-derived features unless explicitly valid. Prefer reproducible pipelines; never tune preprocessing against test performance.

## Validation/acceptance
Compare raw vs processed and verify:
- expected shape/row changes; null patterns; duplicate counts and key uniqueness;
- data types, units, domains, ranges, categories, and cross-field rules;
- distribution changes, excessive clipping/smoothing/imputation;
- join cardinality, match rates, and unmatched keys;
- timestamps, gaps, frame alignment, and keypoint integrity;
- privacy controls; fit-for-purpose status.
Warn/fail on unmet required constraints. Do not claim data is clean without evidence.

## Deliverables and audit trail
As applicable, provide processed dataset(s), a preprocessing report (issues, actions, before/after counts, validation), config/parameters/mappings, rejected rows with reason codes (only if safe/useful), logs and versions. Suggested structure:
\`\`\`
preprocessing_output/
  data/processed_dataset.csv
  reports/preprocessing_report.md
  configs/preprocessing_config.json
  logs/preprocessing.log
  rejected/rejected_records.csv
\`\`\`
Avoid sensitive raw values in reports/logs.

## Agent checklist and reporting
- Define goal/output; preserve originals.
- Profile and verify missing/sentinel rules, anomalies, duplicates, types, units, constraints.
- Validate join keys/cardinality when integrating.
- Justify transformations/scaling/reduction and assess information loss.
- Protect sensitive data; prevent ML leakage.
- Validate and document parameters, row lineage, changes, limitations.

Final report should state: input/scope; key issues; actions and parameters; rows/columns changed/removed/created with reasons; checks/results; output paths; assumptions, limitations, next steps. If nontrivial or destructive operations are proposed, summarize the plan before applying them.
