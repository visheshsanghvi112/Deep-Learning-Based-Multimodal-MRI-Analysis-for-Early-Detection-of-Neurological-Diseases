import os

def list_adni_folders():
    path = r"C:\Users\gener\Downloads"
    try:
        items = os.listdir(path)
        adni = [i for i in items if "ADNI" in i]
        print(f"Found {len(adni)} ADNI items:")
        for a in sorted(adni):
            print(f" - {a}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_adni_folders()
