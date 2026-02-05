import nibabel as nib
import numpy as np
import os

# Test with one file
test_file = r"d:\discs\data\ADNI\002_S_0295\MPR__GradWarp__B1_Correction__N3__Scaled\2006-04-18_08_20_30.0\I45108\ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070319113623975_S13408_I45108.nii"

if os.path.exists(test_file):
    img = nib.load(test_file)
    data = img.get_fdata()
    print(f"Shape: {data.shape}")
    print(f"Affine:\n{img.affine}")
    
    # Test different slicing dimensions
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sagittal (YZ plane, varies along X)
    sag_slice = data[data.shape[0]//2, :, :]
    axes[0].imshow(np.rot90(sag_slice), cmap='gray')
    axes[0].set_title(f'Sagittal: data[{data.shape[0]//2}, :, :]')
    axes[0].axis('off')
    
    # Coronal (XZ plane, varies along Y)
    cor_slice = data[:, data.shape[1]//2, :]
    axes[1].imshow(np.rot90(cor_slice), cmap='gray')
    axes[1].set_title(f'Coronal: data[:, {data.shape[1]//2}, :]')
    axes[1].axis('off')
    
    # Axial (XY plane, varies along Z)
    ax_slice = data[:, :, data.shape[2]//2]
    axes[2].imshow(np.rot90(ax_slice), cmap='gray')
    axes[2].set_title(f'Axial: data[:, :, {data.shape[2]//2}]')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(r'd:\discs\figures\mri_orientation_test.png', dpi=150)
    print("\nTest figure saved to d:\\discs\\figures\\mri_orientation_test.png")
    plt.show()
else:
    print(f"File not found: {test_file}")
