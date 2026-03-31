# Changelog: Study ID and Auto-Selection Features

**Date:** December 17, 2024  
**Version:** 1.1  
**Changes:** Added study_id column and auto-selection for validated RTStruct

---

## Summary of Changes

Two major enhancements to the `generate_study_folder_inventory()` function:

1. **Added `study_id` column** - Unique identifier for grouping series by study
2. **Auto-selection for validated RTStruct** - Automatically marks validated RTStruct with "x"

---

## 1. Study ID Column

### What Was Added

A new **`study_id`** column has been added as the **first column** in the output files.

### Format

```
{patient_id}_regpair{nn}_{study_type_short}_{YYYY-MM}
```

**Where:**
- `regpair{nn}` = Registration pair number **per patient** (resets for each patient)
- Each patient's first pair → regpair01
- Each patient's second pair → regpair02
- etc.

**Examples:**
- `0871_regpair01_initial_1997-07` - Patient 0871, registration pair 1, July 1997, initial scan
- `0871_regpair01_followup_1999-04` - Patient 0871, registration pair 1, April 1999, followup scan
- `0944_regpair01_initial_1998-11` - Patient 0944, registration pair 1 (counter resets!)
- `3126_regpair01_initial_2009-08` - Patient 3126, registration pair 1, August 2009, initial scan
- `3126_regpair02_initial_2009-08` - Patient 3126, registration pair 2, August 2009, initial scan (different pair!)
- `3126_regpair03_initial_2009-12` - Patient 3126, registration pair 3 (third pair for this patient)
- `3126_regpair04_initial_2010-03` - Patient 3126, registration pair 4 (fourth pair for this patient)

### Purpose

The `study_id` serves as a **unique identifier** that:
- **Includes registration pair number** from the input CSV row order
- Groups all DICOM series (MR, RTStruct, RTDose) from the same study together
- **Preserves the exact order** from the registration pair CSV file
- Makes it easier to identify which series belong to the same registration pair
- Handles cases where the same patient/date appears in multiple registration pairs
- Provides a convenient reference for data organization and processing
- Simplifies filtering and grouping in Excel or pandas

### Implementation Details

```python
# Track regpair number per patient (resets for each new patient)
patient_regpair_counter = {}  # {patient_id: counter}

for idx, row in df_pairs.iterrows():
    patient_id = row['patient_id']
    
    # Increment regpair counter for this patient
    if patient_id not in patient_regpair_counter:
        patient_regpair_counter[patient_id] = 1
    else:
        patient_regpair_counter[patient_id] += 1
    
    # Registration pair number for THIS patient (formatted as 01, 02, 03, etc.)
    regpair_num = str(patient_regpair_counter[patient_id]).zfill(2)
    regpair_order = idx * 2  # For sorting: maintains CSV row order
    ...

# Build study_id: patient_regpairNN_type_YYYY-MM
date_folder = path_utils.convert_iso_date_to_folder_format(date_iso)  # YYYY-MM
study_type_short = 'initial' if 'initial' in study_type else 'followup'
study_id = f"{patient_id}_regpair{regpair_num}_{study_type_short}_{date_folder}"
# Example: "3126_regpair02_initial_2009-08"
```

**Key points:**
- Each patient has their own counter starting from 1
- Counter increments each time we see the same patient in CSV
- regpair_order still tracks CSV row sequence for sorting
- Same patient/date combination can appear in multiple regpairs (kept, not deduplicated)

### Where It Appears

- **CSV file**: First column
- **Excel file**: First column (column A)
- **Sorting**: Primary sort key (groups all series from same study together)

---

## 2. Auto-Selection for Validated RTStruct

### What Was Changed

When `validate_rtstruct=True` (default), the function now **automatically marks validated RTStruct series** with "x" in the `selected` column.

### Behavior

**Before:**
- All `selected` columns were empty
- User had to manually review and mark ALL series, including RTStruct

**After:**
- RTStruct that match validation patterns are **pre-marked with "x"**
- All other series (MR, RTDose, non-validated RTStruct) remain empty
- User only needs to manually select MR series and review pre-selections

### Validation Logic

```python
# Check if RTStruct matches expected patterns
is_valid = any(pattern in series_desc for pattern in expected_rtstruct_patterns)

# Auto-select if validated
row_data['selected'] = 'x' if is_valid else ''
```

**Default validation patterns:**
- `Brain_MS_Init_Model` (initial treatment structures)
- `Brain_MS_ReTx_Model` (re-treatment structures)

### Example Output

```
study_id          | modality  | series_description    | selected | rtstruct_validated
------------------|-----------|----------------------|----------|-------------------
0871_1997-07_init | MR        | T1_axial_post_gad    |          | 
0871_1997-07_init | RTSTRUCT  | Brain_MS_Init_Model  | x        | PASS    ← Auto-selected!
0871_1997-07_init | RTSTRUCT  | Backup_Structures    |          |         ← Not selected
0871_1997-07_init | RTDOSE    | Treatment_Dose       |          | 
```

### User Workflow Impact

**New workflow:**
1. ✅ Run notebook to generate inventory
2. ✅ Open Excel file
3. ✅ **RTStruct already selected** for validated patterns (marked with "x")
4. 🔍 Review pre-selected RTStruct (optional - change if needed)
5. ✏️ **Only need to manually select MR series** (one per study)
6. ✏️ Manually select RTDose if needed
7. ✅ Save Excel file

**Benefits:**
- Saves time - no need to search for correct RTStruct
- Reduces errors - validation ensures correct naming patterns
- Still flexible - user can override by changing selections

---

## Updated Column Order

### New Column Order (CSV and Excel)

1. **`study_id`** ⭐ NEW
2. `patient_id`
3. `study_date`
4. `study_type`
5. `targets`
6. `n_targets`
7. `modality`
8. `folder_name`
9. `series_description`
10. `file_count`
11. `selected` ⭐ MODIFIED (auto-populated for validated RTStruct)
12. `rtstruct_validated` (optional, if validate_rtstruct=True)
13. `image_dimensions` (optional, if include_metadata=True)
14. `orientation` (optional, if include_metadata=True)
15. `acquisition_date` (optional, if include_metadata=True)

---

## Sorting Changes

### Previous Sorting

```python
df_inventory.sort_values(
    by=['patient_id', 'study_date', 'study_type', 'modality', 'folder_name']
)
```

### New Sorting

```python
df_inventory.sort_values(
    by=['regpair_order', 'modality', 'folder_name']
)
# regpair_order column is then dropped after sorting
```

**NEW BEHAVIOR: Follows CSV row order exactly!**
- Row 1 in CSV → all series appear first
- Row 2 in CSV → all series appear second
- etc.
- Within each registration pair: initial study first, then followup study
- Within each study: MR → RTSTRUCT → RTDOSE (alphabetical by modality)

**Example order from CSV:**
```
CSV Row 1: 0871, 1997-07-18 → 1999-04-02  (0871's 1st pair)
CSV Row 2: 0944, 1998-11-12 → 2000-07-28  (0944's 1st pair)
CSV Row 3: 0979, 1998-11-23 → 1999-08-11  (0979's 1st pair)
CSV Row 4: 0979, 1998-11-23 → 2000-03-22  (0979's 2nd pair)
...
CSV Row 30: 3126, 2009-08-28 → 2010-03-18  (3126's 1st pair)
CSV Row 31: 3126, 2009-08-28 → 2011-01-07  (3126's 2nd pair - same date!)
CSV Row 32: 3126, 2009-12-23 → 2011-01-07  (3126's 3rd pair)
CSV Row 33: 3126, 2010-03-18 → 2011-01-07  (3126's 4th pair)
```

**Output order:**
```
0871_regpair01_initial_1997-07   (all MR, RTStruct, RTDose)
0871_regpair01_followup_1999-04  (all MR, RTStruct, RTDose)
0944_regpair01_initial_1998-11   (all MR, RTStruct, RTDose) ← Counter resets for 0944
0944_regpair01_followup_2000-07  (all MR, RTStruct, RTDose)
0979_regpair01_initial_1998-11   (all MR, RTStruct, RTDose) ← Counter resets for 0979
0979_regpair01_followup_1999-08  (all MR, RTStruct, RTDose)
0979_regpair02_initial_1998-11   (all MR, RTStruct, RTDose) ← 0979's 2nd pair
0979_regpair02_followup_2000-03  (all MR, RTStruct, RTDose)
...
3126_regpair01_initial_2009-08   (all MR, RTStruct, RTDose) ← Counter resets for 3126
3126_regpair01_followup_2010-03  (all MR, RTStruct, RTDose)
3126_regpair02_initial_2009-08   (all MR, RTStruct, RTDose) ← Same date, different pair!
3126_regpair02_followup_2011-01  (all MR, RTStruct, RTDose)
3126_regpair03_initial_2009-12   (all MR, RTStruct, RTDose) ← 3rd pair
3126_regpair03_followup_2011-01  (all MR, RTStruct, RTDose)
3126_regpair04_initial_2010-03   (all MR, RTStruct, RTDose) ← 4th pair
3126_regpair04_followup_2011-01  (all MR, RTStruct, RTDose)
```

**Benefits:**
- **Preserves input CSV order** - critical for workflow consistency
- Handles duplicate patient/date combinations correctly
- regpair number makes it explicit which registration pair each study belongs to
- Easy to track progress through the CSV file

---

## Configuration

### No Changes Required

The new features work with existing configuration:

```python
# These settings control the new features
validate_rtstruct = True        # Enable auto-selection (default: True)
expected_rtstruct_patterns = [  # Patterns to validate
    "Brain_MS_Init_Model",
    "Brain_MS_ReTx_Model"
]
```

**To disable auto-selection:**
```python
validate_rtstruct = False  # Disables both validation column AND auto-selection
```

---

## Backward Compatibility

### CSV/Excel Files

- **Old scripts reading the CSV/Excel**: May need to adjust column indices
  - `study_id` is now column 0 (first column)
  - All other columns shifted by +1 position
  
- **Recommended**: Use column names instead of indices when reading files
  ```python
  df = pd.read_csv('inventory.csv')
  patient_ids = df['patient_id']  # ✅ Works regardless of column order
  ```

### Function Signature

- **No changes** to function parameters
- All existing code calling `generate_study_folder_inventory()` works as-is
- New behavior is automatic based on existing `validate_rtstruct` parameter

---

## Examples

### Example 1: Basic Output (Counter Resets Per Patient)

```
study_id                       | patient_id | study_date  | modality  | series_description    | selected
-------------------------------|------------|-------------|-----------|----------------------|----------
0871_regpair01_initial_1997-07 | 0871       | 1997-07-18  | MR        | T1_axial_post_gad    |           ← 0871's 1st pair
0871_regpair01_initial_1997-07 | 0871       | 1997-07-18  | RTSTRUCT  | Brain_MS_Init_Model  | x
0871_regpair01_followup_1999-04| 0871       | 1999-04-02  | MR        | T1_axial_post_gad    |
0871_regpair01_followup_1999-04| 0871       | 1999-04-02  | RTSTRUCT  | Brain_MS_ReTx_Model  | x
0944_regpair01_initial_1998-11 | 0944       | 1998-11-12  | MR        | T1_axial_post_gad    |           ← 0944's 1st pair (counter reset!)
0944_regpair01_followup_2000-07| 0944       | 2000-07-28  | MR        | T1_axial_post_gad    |
0979_regpair01_initial_1998-11 | 0979       | 1998-11-23  | MR        | T1_axial_post_gad    |           ← 0979's 1st pair (counter reset!)
0979_regpair02_initial_1998-11 | 0979       | 1998-11-23  | MR        | T1_axial_post_gad    |           ← 0979's 2nd pair (same patient!)
```

### Example 2: Multiple RTStruct Per Study

```
study_id                       | modality  | series_description      | selected | rtstruct_validated
-------------------------------|-----------|------------------------|----------|-------------------
1885_regpair26_initial_2004-01 | RTSTRUCT  | Brain_MS_Init_Model    | x        | PASS    ← Auto-selected
1885_regpair26_initial_2004-01 | RTSTRUCT  | Backup_Structures      |          |         ← Not selected
1885_regpair26_initial_2004-01 | RTSTRUCT  | Old_Contours_Archive   |          |         ← Not selected
```

Only the validated one is auto-selected!

### Example 3: Same Patient, Multiple Registration Pairs

For patient 3126 with 4 registration pairs (from CSV rows 30-33):

```
study_id                       | patient_id | study_date  | targets     | modality
-------------------------------|------------|-------------|-------------|----------
3126_regpair01_initial_2009-08 | 3126       | 2009-08-28  | 02/05       | MR        ← 3126's 1st pair
3126_regpair01_followup_2010-03| 3126       | 2010-03-18  | 02/05       | MR
3126_regpair02_initial_2009-08 | 3126       | 2009-08-28  | 06/07/08/.. | MR        ← 3126's 2nd pair, same date!
3126_regpair02_followup_2011-01| 3126       | 2011-01-07  | 06/07/08/.. | MR
3126_regpair03_initial_2009-12 | 3126       | 2009-12-23  | 13/14/15    | MR        ← 3126's 3rd pair
3126_regpair03_followup_2011-01| 3126       | 2011-01-07  | 13/14/15    | MR        ← Same date as pair 2!
3126_regpair04_initial_2010-03 | 3126       | 2010-03-18  | 18/19/21    | MR        ← 3126's 4th pair, same date as pair 1!
3126_regpair04_followup_2011-01| 3126       | 2011-01-07  | 18/19/21    | MR        ← Same date as pairs 2 & 3!
```

Notice:
- Counter is per-patient: regpair01, regpair02, regpair03, regpair04 (all for patient 3126)
- Multiple pairs can share same dates (2009-08-28 used in pairs 1 & 2, 2011-01-07 used in pairs 2, 3, & 4)
- Different targets distinguish the pairs (02/05 vs 06/07/08/11/12 vs 13/14/15 vs 18/19/21)
- Each gets a unique study_id with its per-patient regpair number

### Example 4: Using study_id for Filtering

```python
# Read the inventory
df = pd.read_csv('prep1_step1_study_folder_inventory.csv')

# Filter by specific study
study_0871_init = df[df['study_id'] == '0871_regpair01_initial_1997-07']

# Get all selected series for a study
selected = df[(df['study_id'] == '0871_regpair01_initial_1997-07') & (df['selected'] == 'x')]

# Group by study_id
for study_id, group in df.groupby('study_id'):
    print(f"Study {study_id}: {len(group)} series")

# Filter by registration pair number
regpair01 = df[df['study_id'].str.contains('_regpair01_')]

# Filter by patient and regpair
patient_3126_regpair30 = df[df['study_id'].str.startswith('3126_regpair30_')]
```

---

## Documentation Updates

### Updated Files

- ✅ `data_analysis.py` - Function docstring updated
  - Added `study_id` column description
  - Updated `selected` column description
  - Updated sorting description
  - Updated example rows
  - Updated Notes section

### Need to Update (Recommended)

- [ ] `README.md` - Add study_id column to function description
- [ ] `IMPLEMENTATION_SUMMARY.md` - Update example output
- [ ] `STUDY_INVENTORY_USAGE_GUIDE.md` - Update examples and tips
- [ ] `IMPLEMENTATION_COMPLETE.md` - Update example tables

---

## Testing Checklist

Before using in production:

- [ ] Run notebook with `validate_rtstruct=True`
- [ ] Verify `study_id` appears as first column
- [ ] Verify validated RTStruct have "x" in `selected` column
- [ ] Verify non-validated RTStruct do NOT have "x"
- [ ] Verify MR and RTDose do NOT have "x" (empty)
- [ ] Verify sorting groups series by study_id
- [ ] Verify CSV and Excel have same data
- [ ] Test with `validate_rtstruct=False` (no auto-selection)

---

## Migration Guide

### For Existing Users

**If you have existing inventory files:**

1. **Regenerate** the inventory using the updated function
2. **Column shift**: Be aware that all columns are shifted by +1 (study_id is new first column)
3. **Auto-selection**: Review pre-selected RTStruct, adjust if needed
4. **Update scripts**: If reading CSV/Excel, use column names not indices

**Example script update:**

```python
# ❌ Old way (brittle, breaks with new columns)
patient_id = df.iloc[:, 0]  # Was first column, now second!

# ✅ New way (robust, works with any column order)
patient_id = df['patient_id']
```

---

## Benefits Summary

### For Users

✅ **Faster QC** - RTStruct pre-selected, only need to choose MR  
✅ **Better organization** - study_id groups related series  
✅ **Fewer errors** - Validated RTStruct automatically identified  
✅ **Easier filtering** - study_id simplifies Excel filtering  
✅ **Better documentation** - Clear which series belong together  

### For Developers

✅ **Consistent identifiers** - study_id provides stable reference  
✅ **Simpler grouping** - Single column instead of multiple  
✅ **Better UX** - Users appreciate auto-selection feature  
✅ **Backward compatible** - Function signature unchanged  

---

**Implementation completed:** December 17, 2024  
**Updated:** December 18, 2024 - Added regpair column to Part A output  
**Tested:** Pending user validation  
**Follows:** coding_guide.md standards

---

## Bug Fix: Leading Zero Preservation (December 17, 2024)

### Issue
patient_id was being read from CSV as integer, causing leading zeros to be lost:
- Expected: `0871_regpair01_initial_1997-07`
- Actual: `871_regpair01_initial_1997-07` ❌

### Solution
Added two safeguards:

1. **Read CSV with dtype specification:**
```python
df_pairs = pd.read_csv(csv_path, dtype={'patient_id': str})
```

2. **Ensure zero-padding:**
```python
patient_id = str(row['patient_id']).zfill(4)  # Always 4 digits
```

### Result
✅ patient_id: `0871`, `0944`, `0979`, `1036`, etc. (leading zeros preserved)  
✅ study_id: `0871_regpair01_initial_1997-07` (correct format)

---

## Enhancement: Add regpair Column to Part A Output (December 18, 2024)

### Motivation
Previously, the regpair counter was calculated independently in Part B (`generate_study_folder_inventory()`). This created potential inconsistency if the logic diverged. To ensure consistency and follow DRY (Don't Repeat Yourself) principle, the regpair counter should be calculated once in Part A and reused in Part B.

### Changes Made

#### 1. Modified `build_summaries()` in `data_analysis.py`

**Added Step 8:** After creating the per_pair DataFrame, add a regpair counter:
```python
# Step 8: Add regpair counter (resets for each patient)
per_pair['regpair'] = (
    per_pair.groupby('patient_id').cumcount() + 1
).apply(lambda x: str(x).zfill(2))

# Reorder columns: patient_id, regpair, initial_moving, ...
cols = ['patient_id', 'regpair', 'initial_moving', 'followup_fixed', 'targets', 'n_targets']
per_pair = per_pair[cols]
```

**Updated Documentation:**
- Updated OUTPUT FORMAT section to include regpair column description
- Updated example outputs to show regpair column
- Added notes explaining that regpair resets for each patient

#### 2. Modified `generate_study_folder_inventory()` in `data_analysis.py`

**Removed:** Manual calculation of regpair counter using `patient_regpair_counter` dictionary

**Added:** Read regpair directly from CSV:
```python
# Read regpair from CSV (already formatted as '01', '02', ...)
regpair_num = str(row['regpair']).zfill(2)
```

**Updated Documentation:**
- Updated required columns to include 'regpair'
- Added notes that regpair is generated by `build_summaries()`

#### 3. Updated `README.md`

**Modified:**
- Per-Registration-Pair Summary format now includes regpair column
- Updated `build_summaries()` description to mention automatic regpair generation

### Result

**Part A Output (prep1_step1_summary_by_registrationpair.csv):**
```
patient_id,regpair,initial_moving,followup_fixed,targets,n_targets
0871,01,1997-07-18,1999-04-02,01,1
0944,01,1998-11-12,2000-07-28,13,1
0979,01,1998-11-23,1999-08-11,03,1
0979,02,1998-11-23,2000-03-22,02,1
1036,01,1998-10-22,2000-11-28,03,1
3126,01,2009-08-28,2010-03-18,02/05,2
3126,02,2009-08-28,2011-01-07,06/07/08/11/12,5
3126,03,2009-12-23,2011-01-07,13/14/15,3
3126,04,2010-03-18,2011-01-07,18/19/21,3
```

**Part B (generate_study_folder_inventory):**
- Now reads regpair directly from CSV instead of calculating
- Ensures perfect consistency with Part A
- Simplifies code by removing duplicate logic

### Benefits
✅ **Single Source of Truth:** regpair calculated once in `build_summaries()`  
✅ **Consistency:** Part B always uses exact regpair values from Part A  
✅ **Simplified Code:** Removed duplicate counter logic from Part B  
✅ **Better Documentation:** regpair format documented in one central place  
✅ **Easier Debugging:** If regpair logic needs changes, modify only one place

