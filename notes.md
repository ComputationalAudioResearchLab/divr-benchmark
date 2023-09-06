## Problems with existing praat-parselmouth library

- The library wraps the C implementation of PRAAT, and as such
  - debugging any errors is extremely difficult as debugging tools don't work across multiple languages
  - even adding logging/print statements for debugging requires rebuilding the library
- the existing PRAAT code is extremely hard to customize and extend
- it is almost impossible to identify the mapping from mathematical functions to PRAAT code implementation
