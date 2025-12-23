import os

def check_access():
    path = r"C:\Users\gener\Downloads"
    print(f"Checking access to: {path}")
    
    try:
        # Check existence
        if not os.path.exists(path):
            print("Path does not exist!")
            return

        # Check listability
        items = os.listdir(path)
        print(f"Found {len(items)} items in root.")
        
        # Look for ADNI folders
        adni_folders = [i for i in items if "ADNI" in i]
        print("ADNI-related folders found:", adni_folders)
        
        # Deep scan for NIfTI files in one of them to verify read access
        if adni_folders:
            target = os.path.join(path, adni_folders[0])
            print(f"Deep scanning {target} for first 3 .nii files...")
            count = 0
            for root, dirs, files in os.walk(target):
                for f in files:
                    if f.endswith(".nii"):
                        print(f"Found NIfTI: {os.path.join(root, f)}")
                        count += 1
                        if count >= 3:
                            return
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_access()
