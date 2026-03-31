# Coding Guide for RTdicomorganizer

## Purpose
This document defines the coding standards and documentation requirements for the RTdicomorganizer project. **ALL generative models and developers must review and follow these rules when editing code.**

---

## 🔴 BOTTOM LINE RULE

**Every time code is edited:**
1. ✅ Review and follow ALL rules in this `coding_guide.md`
2. ✅ Update the `README.md` file if structure/dependencies change
3. ✅ Update ALL related comments and docstrings
4. ✅ Maintain consistency with existing code style

---

## 📁 Level 1: Folder-Level Documentation

### Requirements for `README.md`:
The folder must contain a `README.md` file that includes:

1. **Folder Structure Overview**
   - Visual tree showing all files and subfolders
   - Brief description of what each file does

2. **Module Dependencies**
   - Clear dependency graph or list showing which modules depend on others
   - Example:
     ```
     visualization.py → dicom_utils.py → path_utils.py
     ```

3. **Complete Function/Class Inventory**
   - List ALL functions and classes from ALL files
   - Organized by file
   - Include one-line description for each function/class

4. **Quick Start Guide**
   - Basic usage examples
   - How to import and use the modules

---

## 📄 Level 2: Python File-Level Documentation

### Requirements at the TOP of each `.py` file:

```python
"""
==============================================================================
MODULE NAME: <module_name.py>
==============================================================================

PURPOSE:
    <Detailed description of what this module does>
    <Explain the role of this module in the larger project>
    <Describe key use cases>

DEPENDENCIES:
    - External packages: numpy, pandas, pathlib, etc.
    - Internal modules: other modules from this package
    
FUNCTIONS/CLASSES:
    1. function_name_1(arg1, arg2) -> return_type
       Brief description of what it does
       
    2. function_name_2(arg1, arg2) -> return_type
       Brief description of what it does
       
    3. ClassName
       Brief description of the class purpose

USAGE EXAMPLE:
    ```python
    from RTdicomorganizer import module_name
    
    result = module_name.function_name_1(arg1, arg2)
    ```

NOTES:
    - Any important caveats or limitations
    - Any assumptions made by the code
    
==============================================================================
"""
```

---

## 🔧 Level 3: Function/Class-Level Documentation

### Requirements for EVERY function:

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    One-line summary of what the function does.
    
    DETAILED DESCRIPTION:
        Multi-line detailed explanation of:
        - What the function does
        - Why it exists
        - How it works (algorithm/process steps)
        - Any important implementation details
    
    Args:
        param1 (type): Description of param1
            - Any constraints or requirements
            - Valid value ranges
            - Default behavior if optional
            
        param2 (type): Description of param2
            - Additional details if complex type
    
    Returns:
        return_type: Description of return value
            - Format of the returned data
            - Any special properties
    
    Raises:
        ExceptionType: When and why this exception is raised
    
    Example:
        >>> result = function_name("value1", 42)
        >>> print(result)
        Expected output
    
    Notes:
        - Any important caveats
        - Edge cases to be aware of
        - Related functions to consider
    """
```

### Special Requirements for CSV/XLSX Handling Functions:

When a function reads or writes CSV/XLSX files, include **detailed format specifications**:

```python
def read_lesion_data(csv_path: str) -> pd.DataFrame:
    """
    Read lesion data from CSV file with specific column format.
    
    INPUT FILE FORMAT:
        Expected CSV columns:
        - patient_id (str): 4-digit patient ID with leading zeros
          Example: '0871', '1885', '3126'
          
        - target (str): 2-digit target/lesion ID with leading zeros
          Example: '01', '09', '15'
          
        - lesion_label (str): Combined format 'PPPP.TT'
          Example: '0871.01', '1885.09', '3126.15'
          
        - initial_moving (str): Initial scan date in ISO format (YYYY-MM-DD)
          Example: '1997-07-18', '2010-10-05'
          
        - followup_fixed (str): Follow-up scan date in ISO format (YYYY-MM-DD)
          Example: '1999-04-02', '2015-11-12'
    
    OUTPUT FORMAT:
        Returns DataFrame with columns:
        - patient_id (str): 4-digit string, zero-padded
        - target (str): 2-digit string, zero-padded
        - lesion_label (str): format 'PPPP.TT'
        - initial_moving (datetime64): parsed datetime
        - followup_fixed (datetime64): parsed datetime
    
    Example:
        >>> df = read_lesion_data('summary_by_lesion.csv')
        >>> print(df.dtypes)
        patient_id         object
        target            object
        lesion_label      object
        initial_moving    datetime64[ns]
        followup_fixed    datetime64[ns]
    """
```

---

## 🎨 Code Style Guidelines

### 1. **Import Organization**
```python
# Standard library imports
import os
import sys
from pathlib import Path
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
from . import path_utils
from . import data_io
```

### 2. **Naming Conventions**
- **Functions**: `lowercase_with_underscores`
- **Classes**: `CamelCase`
- **Constants**: `UPPERCASE_WITH_UNDERSCORES`
- **Private functions**: `_leading_underscore` (but avoid if function is generally useful)

### 3. **Type Hints**
Always include type hints for function parameters and return values:
```python
def process_data(input_path: str, threshold: float = 0.5) -> pd.DataFrame:
    pass
```

### 4. **Error Handling**
- Use descriptive error messages
- Include context about what failed
```python
if not path.exists():
    raise FileNotFoundError(
        f"Input file not found: {path}\n"
        f"Please ensure the file exists and path is correct."
    )
```

### 5. **Constants and Configuration**
- Define constants at module level
- Use ALL_CAPS for constant names
- Group related constants together

---

## 📊 Data Format Documentation Standards

### For DataFrame-based functions:
Always document:
1. Expected column names
2. Data types for each column
3. Valid value ranges or formats
4. Any required preprocessing
5. Output format if transformed

### For File-based functions:
Always document:
1. Expected file format (CSV, XLSX, JSON, etc.)
2. Required columns/fields
3. Optional columns/fields
4. Example rows
5. Output file format if different from input

---

## 🧪 Testing Considerations

While not required in this initial implementation, consider:
- Edge cases (empty DataFrames, missing files, etc.)
- Input validation
- Clear error messages for common mistakes

---

## 📝 Documentation Maintenance

### When to Update Documentation:

1. **Adding a new function**:
   - Update file-level function list
   - Update README.md function inventory
   - Add complete docstring with examples

2. **Modifying a function**:
   - Update docstring if signature changes
   - Update examples if behavior changes
   - Update README.md if description changes

3. **Changing dependencies**:
   - Update file-level dependency list
   - Update README.md dependency graph
   - Update imports in affected files

4. **Changing data formats**:
   - Update INPUT/OUTPUT format documentation
   - Update examples
   - Add migration notes if breaking change

---

## ✅ Checklist for Code Reviews

Before committing code, verify:
- [ ] All functions have complete docstrings
- [ ] CSV/XLSX functions have format documentation
- [ ] File-level documentation is updated
- [ ] README.md reflects current state
- [ ] Type hints are present
- [ ] Error messages are descriptive
- [ ] Examples are accurate and tested
- [ ] Dependencies are documented

---

## 🔄 Version History

- **v1.0** (2024-12-17): Initial coding guide created
  - Established 3-level documentation structure
  - Defined CSV/XLSX documentation requirements
  - Set code style guidelines

---

**Last Updated:** 2024-12-17  
**Maintainer:** RTdicomorganizer Team

