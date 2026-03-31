"""
Common DICOM header type mapping for mydicomreader package.

This module contains a type mapping dictionary that maps DICOM Value Representation (VR)
codes to human-readable type descriptions. This is used across different reader classes
to provide consistent metadata display.
"""

# Mapping of DICOM Value Representation (VR) codes to human-readable type descriptions
type_map = {
    'CS': 'String',             # Code String (controlled vocabulary-ish, typically uppercase)
    'SH': 'String',             # Short String (<= 16 chars)
    'LO': 'String',             # Long String (longer free text, <= 64 chars)
    'ST': 'String',             # Short Text (multi-line text, <= 1024 chars)
    'LT': 'String',             # Long Text (multi-line text, <= 10240 chars)
    'UT': 'String',             # Unlimited Text (very long free text)
    'PN': 'String',             # Person Name (formatted name components)
    'UI': 'UID',                # Unique Identifier (UID string like "1.2.840....")

    'DA': 'Date',               # Date (YYYYMMDD)
    'TM': 'Time',               # Time (HHMMSS.frac)
    'DT': 'DateTime',           # DateTime (YYYYMMDDHHMMSS.frac&timezone)

    'IS': 'Integer (as String)',# Integer String (integer stored as ASCII text)
    'DS': 'Number (as String)', # Decimal String (decimal number stored as ASCII text)

    'SS': 'Integer',            # Signed Short (16-bit integer)
    'US': 'Integer',            # Unsigned Short (16-bit integer)
    'SL': 'Integer',            # Signed Long (32-bit integer)
    'UL': 'Integer',            # Unsigned Long (32-bit integer)

    'FL': 'Float',              # Floating Point Single (32-bit float)
    'FD': 'Float',              # Floating Point Double (64-bit float)

    'OB': 'Binary',             # Other Byte (raw bytes)
    'OW': 'Binary',             # Other Word (16-bit words; often pixel data)
    'OF': 'Binary',             # Other Float (32-bit float array)
    'OD': 'Binary',             # Other Double (64-bit float array)

    'SQ': 'Sequence',           # Sequence of items (nested datasets)
    'AT': 'Tag'                 # Attribute Tag (stores a DICOM tag like (gggg,eeee))
}

