"""
Complete data compatibility check for OASIS dataset
Scans disc1 root + disc2-12 subfolders
Handles both n3 and n4 MRI file variants
"""
import pandas as pd
import os
import nibabel as nib

BASE = 'D:/disc1'

def get_all_subject_paths():
    """Get all subject folder paths across all discs"""
    paths = {}
    
    # disc1 root
    for f in os.listdir(BASE):
        if f.startswith('OAS1_') and os.path.isdir(f'{BASE}/{f}'):
            paths[f] = f'{BASE}/{f}'
    
    # disc2-12 subfolders
    for i in range(2, 13):
        disc_path = f'{BASE}/disc{i}'
        if os.path.exists(disc_path):
            for f in os.listdir(disc_path):
                if f.startswith('OAS1_') and os.path.isdir(f'{disc_path}/{f}'):
                    paths[f] = f'{disc_path}/{f}'
    
    return paths


def find_mri_file(subj_id, subj_path):
    """Find MRI file - handles both n3 and n4 variants"""
    t88_path = f'{subj_path}/PROCESSED/MPRAGE/T88_111'
    if not os.path.exists(t88_path):
        return None, None
    
    # Try n4 first (newer), then n3 (older)
    for variant in ['n4', 'n3']:
        mri_file = f'{t88_path}/{subj_id}_mpr_{variant}_anon_111_t88_masked_gfc.hdr'
        if os.path.exists(mri_file):
            return mri_file, variant
    
    return None, None


def main():
    print('='*70)
    print('OASIS FULL DATA COMPATIBILITY CHECK')
    print('='*70)
    
    # Get all subject paths
    subject_paths = get_all_subject_paths()
    print(f'\n[1] MRI FOLDERS FOUND: {len(subject_paths)}')
    
    # Count per disc
    disc_counts = {}
    for subj_id, path in subject_paths.items():
        parts = path.split('/')
        disc = 'disc1'
        for p in parts:
            if p.startswith('disc') and p != 'disc1':
                disc = p
                break
        disc_counts[disc] = disc_counts.get(disc, 0) + 1
    
    for disc in sorted(disc_counts.keys(), key=lambda x: int(x.replace('disc', ''))):
        print(f'    {disc}: {disc_counts[disc]} subjects')
    
    # Check MRI file availability
    print(f'\n[2] MRI FILE AVAILABILITY:')
    n3_subjects = []
    n4_subjects = []
    missing_subjects = []
    
    for subj_id, subj_path in subject_paths.items():
        mri_file, variant = find_mri_file(subj_id, subj_path)
        if variant == 'n4':
            n4_subjects.append(subj_id)
        elif variant == 'n3':
            n3_subjects.append(subj_id)
        else:
            missing_subjects.append(subj_id)
    
    print(f'    N4 bias correction: {len(n4_subjects)} subjects')
    print(f'    N3 bias correction: {len(n3_subjects)} subjects')
    print(f'    Missing MRI file:   {len(missing_subjects)} subjects')
    print(f'    TOTAL WITH MRI:     {len(n4_subjects) + len(n3_subjects)}')
    
    if missing_subjects:
        print(f'    Missing: {missing_subjects}')
    
    # Load metadata
    df = pd.read_excel(f'{BASE}/oasis_cross-sectional-5708aa0a98d82080.xlsx')
    print(f'\n[3] METADATA ENTRIES: {len(df)}')
    
    # Match - only subjects with MRI
    subjects_with_mri = n4_subjects + n3_subjects
    matched = df[df['ID'].isin(subjects_with_mri)]
    print(f'\n[4] MATCHED (MRI + metadata): {len(matched)}')
    
    # CDR distribution
    print(f'\n[5] CDR DISTRIBUTION:')
    cdr_counts = matched['CDR'].value_counts().sort_index()
    for cdr_val, count in cdr_counts.items():
        label = {0: 'Normal', 0.5: 'Very Mild', 1: 'Mild', 2: 'Moderate'}.get(cdr_val, 'Unknown')
        print(f'    CDR={cdr_val}: {count:3d} ({label})')
    
    no_cdr = matched['CDR'].isna().sum()
    print(f'    No CDR: {no_cdr:3d} (healthy controls)')
    
    # Target classes
    cdr_0 = len(matched[matched['CDR'] == 0])
    cdr_05 = len(matched[matched['CDR'] == 0.5])
    
    print(f'\n[6] EARLY DETECTION TARGET:')
    print(f'    CDR=0 (Normal):      {cdr_0} subjects')
    print(f'    CDR=0.5 (Very Mild): {cdr_05} subjects')
    print(f'    TOTAL:               {cdr_0 + cdr_05} subjects')
    
    # Test MRI loading
    print('\n' + '='*70)
    print('TESTING MRI FILE LOADING')
    print('='*70)
    
    import random
    test_subjects = random.sample(subjects_with_mri, min(5, len(subjects_with_mri)))
    
    all_ok = True
    for subj_id in test_subjects:
        subj_path = subject_paths[subj_id]
        mri_file, variant = find_mri_file(subj_id, subj_path)
        try:
            img = nib.load(mri_file)
            print(f'{subj_id} ({variant}): OK - shape {img.shape[:3]}')
        except Exception as e:
            print(f'{subj_id}: ERROR - {e}')
            all_ok = False
    
    # Final summary
    print('\n' + '='*70)
    print('FINAL SUMMARY')
    print('='*70)
    print(f'Total MRI folders:           {len(subject_paths)}')
    print(f'With valid MRI file:         {len(subjects_with_mri)}')
    print(f'With metadata match:         {len(matched)}')
    print(f'For early detection (0/0.5): {cdr_0 + cdr_05}')
    print(f'MRI loading test:            {"PASSED" if all_ok else "FAILED"}')
    
    if all_ok and len(matched) >= 430:
        print('\n*** FULL DATASET READY FOR ANALYSIS! ***')
    
    return subject_paths, subjects_with_mri


if __name__ == '__main__':
    paths, valid_subjects = main()
