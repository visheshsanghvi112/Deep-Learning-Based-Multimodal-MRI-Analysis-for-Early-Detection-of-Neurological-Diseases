# ADNI DATASET COMPREHENSIVE ANALYSIS REPORT
================================================================================

**Generated:** 2025-12-13 16:52:38
**Analysis Base Path:** `D:\discs\ADNI`
**Analysis Type:** Professional Research-Level Deep Analysis

---

## 1. EXECUTIVE SUMMARY
--------------------------------------------------------------------------------

- **Total Files:** 230
- **Total Size:** 7.84 GB
- **NIfTI Files:** 230
- **Subject Folders:** 203
- **Unique Subjects:** 203
- **Scan Types Identified:** 4
- **Files Analyzed (Metadata):** 100
- **Errors Encountered:** 0

---

## 2. FILE TYPES AND EXTENSIONS
--------------------------------------------------------------------------------

| Extension | Count | Percentage |
|-----------|-------|------------|
| `.nii` | 230 | 100.00% |

---

## 3. FOLDER STRUCTURE ANALYSIS
--------------------------------------------------------------------------------

- **Maximum Depth:** 4 levels

### Depth Distribution:
| Depth | File Count |
|-------|------------|
| 4 | 230 |

### Folder Pattern Analysis:
- **Date_Folder:** 230 occurrences
- **ImageID_Folder:** 230 occurrences

### Acquisition Date Range:
- **Earliest Scan:** 2005-09-01
- **Latest Scan:** 2008-09-17
- **Time Span:** 1112 days (3.0 years)

### Scans by Year:
| Year | Count |
|------|-------|
| 2005 | 8 |
| 2006 | 92 |
| 2007 | 111 |
| 2008 | 19 |

### Typical Folder Structure:
```
002_S_0295\MPR__GradWarp__B1_Correction__N3__Scaled\2006-11-02_08_16_44.0\I40966\ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070219173850420_S21856_I40966.nii
```

**Structure Breakdown:**
1. **Subject ID Folder** (e.g., `002_S_0295`)
2. **Scan Type Folder** (e.g., `MPR__GradWarp__B1_Correction__N3__Scaled`)
3. **Date-Time Folder** (e.g., `2006-11-02_08_16_44.0`)
4. **Image ID Folder** (e.g., `I40966`)
5. **NIfTI File** (e.g., `ADNI_002_S_0295_MR_...nii`)

---

## 4. SUBJECT DISTRIBUTION
--------------------------------------------------------------------------------

- **Total Subject Folders:** 203
- **Unique Subject IDs (from filenames):** 203

### Scans per Subject Distribution:
| Scans per Subject | Number of Subjects |
|-------------------|---------------------|
| 1 | 176 |
| 2 | 27 |

---

## 5. SCAN TYPE ANALYSIS
--------------------------------------------------------------------------------

| Scan Type | Count | Percentage |
|-----------|-------|------------|
| `MPR_GradWarp_B1_N3_Scaled` | 125 | 54.35% |
| `MPR-R` | 50 | 21.74% |
| `MPR_N3_Scaled` | 35 | 15.22% |
| `MPR_Other` | 20 | 8.70% |

### Scan Type Descriptions:
- **MPR-R**: Magnetization Prepared Rapid Gradient Echo - Reverse
- **MPR_N3_Scaled**: MPRAGE with N3 bias field correction and scaling
- **MPR_GradWarp_B1_N3_Scaled**: MPRAGE with gradient distortion correction, B1 field correction, N3 bias correction, and scaling

---

## 6. NIFTI FILE METADATA ANALYSIS
--------------------------------------------------------------------------------

**Files Analyzed:** 100

### Image Dimensions (Shape):
| Shape | Count |
|-------|-------|
| `(192, 192, 160)` | 38 |
| `(256, 256, 170)` | 25 |
| `(256, 256, 166)` | 23 |
| `(256, 256, 184)` | 9 |
| `(256, 256, 180)` | 5 |

### Voxel Dimensions:
| Voxel Size (mm) | Count |
|-----------------|-------|
| `0.95 × 0.94 × 1.20` | 1 |
| `0.95 × 0.95 × 1.20` | 1 |
| `0.95 × 0.94 × 1.20` | 1 |
| `0.95 × 0.94 × 1.20` | 1 |
| `0.95 × 0.94 × 1.20` | 1 |
| `0.95 × 0.94 × 1.21` | 1 |
| `1.26 × 1.25 × 1.20` | 1 |
| `1.26 × 1.25 × 1.20` | 1 |
| `0.96 × 0.95 × 1.21` | 1 |
| `0.94 × 0.94 × 1.20` | 1 |

### Data Types:
| Data Type | Count |
|-----------|-------|
| `float64` | 100 |

### Intensity Statistics:
- **Mean Intensity (across files):** 140.12 ± 56.00
- **Std Intensity (across files):** 210.40 ± 85.32
- **Brain Percentage (non-zero voxels):** 65.68% ± 16.54%

### Sample File Metadata:

<details>
<summary>Click to expand sample metadata (first 5 files)</summary>

```

// File 1: ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070219173850420_S21856_I40966.nii
{
  "file": "ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070219173850420_S21856_I40966.nii",
  "path": "002_S_0295\\MPR__GradWarp__B1_Correction__N3__Scaled\\2006-11-02_08_16_44.0\\I40966\\ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070219173850420_S21856_I40966.nii",
  "shape": [
    256,
    256,
    166
  ],
  "dtype": "float64",
  "voxel_dims": [
    0.9474243521690369,
    0.9429909586906433,
    1.201568365097046
  ],
  "affine_shape": [
    4,
    4
  ],
  "data_min": 0.0,
  "data_max": 2685.667724609375,
  "data_mean": 234.4055992255418,
  "data_std": 360.48141064267196,
  "non_zero_voxels": 5674712,
  "total_voxels": 10878976,
  "brain_percentage": 52.16218879423946,
  "header_class": "Nifti1Header"
}

// File 2: ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070713121420365_S32938_I60008.nii
{
  "file": "ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070713121420365_S32938_I60008.nii",
  "path": "002_S_0413\\MPR__GradWarp__B1_Correction__N3__Scaled\\2007-06-01_07_04_09.0\\I60008\\ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070713121420365_S32938_I60008.nii",
  "shape": [
    256,
    256,
    166
  ],
  "dtype": "float64",
  "voxel_dims": [
    0.9490050077438354,
    0.9454396963119507,
    1.2017478942871094
  ],
  "affine_shape": [
    4,
    4
  ],
  "data_min": 0.0,
  "data_max": 2337.30224609375,
  "data_mean": 192.7994625269877,
  "data_std": 321.8483701760927,
  "non_zero_voxels": 5150440,
  "total_voxels": 10878976,
  "brain_percentage": 47.343058758471386,
  "header_class": "Nifti1Header"
}

// File 3: ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001120813046_S22557_I118695.nii
{
  "file": "ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001120813046_S22557_I118695.nii",
  "path": "002_S_0413\\MPR__GradWarp__B1_Correction__N3__Scaled_2\\2006-11-15_09_30_01.0\\I118695\\ADNI_002_S_0413_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001120813046_S22557_I118695.nii",
  "shape": [
    256,
    256,
    166
  ],
  "dtype": "float64",
  "voxel_dims": [
    0.9477056264877319,
    0.9375,
    1.2022950649261475
  ],
  "affine_shape": [
    4,
    4
  ],
  "data_min": 0.0,
  "data_max": 2501.173828125,
  "data_mean": 193.85480164127162,
  "data_std": 323.0866052838287,
  "non_zero_voxels": 5263670,
  "total_voxels": 10878976,
  "brain_percentage": 48.38387362928276,
  "header_class": "Nifti1Header"
}

// File 4: ADNI_002_S_0729_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001115616920_S16874_I118682.nii
{
  "file": "ADNI_002_S_0729_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001115616920_S16874_I118682.nii",
  "path": "002_S_0729\\MPR__GradWarp__B1_Correction__N3__Scaled_2\\2006-07-17_13_27_14.0\\I118682\\ADNI_002_S_0729_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001115616920_S16874_I118682.nii",
  "shape": [
    256,
    256,
    166
  ],
  "dtype": "float64",
  "voxel_dims": [
    0.9475687742233276,
    0.9375,
    1.202091097831726
  ],
  "affine_shape": [
    4,
    4
  ],
  "data_min": 0.0,
  "data_max": 2339.686279296875,
  "data_mean": 183.20053056551816,
  "data_std": 307.9728280952499,
  "non_zero_voxels": 4940878,
  "total_voxels": 10878976,
  "brain_percentage": 45.41675613587161,
  "header_class": "Nifti1Header"
}

// File 5: ADNI_002_S_1018_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070913150134327_S35097_I73016.nii
{
  "file": "ADNI_002_S_1018_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070913150134327_S35097_I73016.nii",
  "path": "002_S_1018\\MPR__GradWarp__B1_Correction__N3__Scaled\\2007-07-16_06_56_24.0\\I73016\\ADNI_002_S_1018_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070913150134327_S35097_I73016.nii",
  "shape": [
    256,
    256,
    166
  ],
  "dtype": "float64",
  "voxel_dims": [
    0.9496818780899048,
    0.9438337683677673,
    1.2032819986343384
  ],
  "affine_shape": [
    4,
    4
  ],
  "data_min": 0.0,
  "data_max": 2689.506103515625,
  "data_mean": 194.15540316323754,
  "data_std": 343.28282060205237,
  "non_zero_voxels": 3961232,
  "total_voxels": 10878976,
  "brain_percentage": 36.41180934676205,
  "header_class": "Nifti1Header"
}
```
</details>

---

## 7. IMAGE FILES ANALYSIS
--------------------------------------------------------------------------------

- **Total Image Files Found:** 0

**No standard image files (PNG, JPG, etc.) found in ADNI folder.**
All imaging data is stored in NIfTI format (.nii files).

---

## 8. NAMING CONVENTION ANALYSIS
--------------------------------------------------------------------------------

### Example Filename:
```
ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070219173850420_S21856_I40966.nii
```

### Filename Structure Breakdown:
The ADNI filename follows this pattern:
```
ADNI_[SubjectID]_MR_[ScanType]_Br_[Date]_S[SeriesID]_I[ImageID].nii
```

**Components:**
- `ADNI`: Dataset identifier
- `[SubjectID]`: Format `XXX_S_XXXX` (e.g., `002_S_0295`)
- `MR`: Modality (Magnetic Resonance)
- `[ScanType]`: Processing pipeline identifier
- `Br`: Brain
- `[Date]`: Processing date (YYYYMMDDHHMMSS)
- `S[SeriesID]`: Series identifier
- `I[ImageID]`: Image identifier

---

## 9. DATA QUALITY ASSESSMENT
--------------------------------------------------------------------------------

### Consistency Checks:
- **Shape Consistency:** [WARN] 5 different shapes found
- **Voxel Size Consistency:** [WARN] 100 different voxel sizes found

### Potential Issues:
- [WARN] High voxel size variability - may need spatial normalization

---

## 10. RESEARCH IMPLICATIONS AND RECOMMENDATIONS
--------------------------------------------------------------------------------

### Dataset Characteristics:
- This ADNI dataset contains **230 structural MRI scans** from **203 unique subjects**
- All data is in **NIfTI format** (.nii), which is standard for neuroimaging research
- Multiple scan types indicate different preprocessing pipelines applied

### Recommended Preprocessing Steps:
1. **Spatial Normalization**: Register all images to a common space (e.g., MNI152)
2. **Resampling**: Ensure consistent voxel sizes across all images
3. **Skull Stripping**: Verify brain extraction (may already be done)
4. **Bias Field Correction**: Some scans have N3 correction, ensure consistency

### Analysis Considerations:
- **Multi-site data**: ADNI data comes from multiple sites - account for site effects
- **Longitudinal potential**: Check if multiple timepoints exist per subject
- **Clinical metadata**: ADNI provides extensive clinical data - link using Subject IDs
- **Quality control**: Review images for artifacts, motion, or acquisition issues

### Compatible Tools:
- **Python**: nibabel, nilearn, scipy, scikit-learn
- **FSL**: fslmaths, flirt, fnirt for preprocessing
- **SPM**: Statistical Parametric Mapping
- **ANTs**: Advanced Normalization Tools
- **FreeSurfer**: Cortical surface analysis

---

## APPENDIX: TECHNICAL DETAILS
--------------------------------------------------------------------------------

### Analysis Parameters:
- Base Path: `D:\discs\ADNI`
- Files Scanned: 230
- NIfTI Files Found: 230
- Metadata Samples Analyzed: 100
- Analysis Date: 2025-12-13T16:52:38.086235

### Software Versions:
- nibabel: 5.3.3
- numpy: 1.26.4

---


**END OF REPORT**

*Report generated by ADNI Comprehensive Analysis Script*
*For questions or issues, review the analysis script: `adni_comprehensive_analysis.py`*