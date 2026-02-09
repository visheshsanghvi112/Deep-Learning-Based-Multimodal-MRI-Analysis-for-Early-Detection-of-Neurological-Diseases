#!/usr/bin/env python3
"""
Cross-verify all numerical metrics between README.md and PROJECT_DOCUMENTATION.md

This script ensures the "product documentation" (source of truth) matches the README.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

def extract_metrics(content: str, context: str = "") -> Dict[str, str]:
    """Extract numerical metrics from markdown content."""
    metrics = {}
    
    # Common patterns to look for
    patterns = {
        'oasis_late_fusion_auc': r'Late\s+Fusion.*?(\d+\.\d+)',
        'oasis_attention_auc': r'Attention\s+Fusion.*?(\d+\.\d+)',
        'oasis_mri_only_auc': r'MRI-Only.*?(\d+\.\d+)',
        'adni_level1_fusion_auc': r'Level-1.*?Fusion.*?(\d+\.\d+)',
        'adni_levelmax_fusion_auc': r'Level-MAX.*?(\d+\.\d+)',
        'longitudinal_auc': r'Longitudinal.*?(\d+\.\d+)',
        'oasis_subjects': r'OASIS.*?(\d+)\s+(?:subjects|usable)',
        'adni_subjects': r'ADNI.*?N=(\d+)',
        'total_scans_longitudinal': r'(\d+,?\d*)\s+(?:total\s+)?scans',
    }
    
    for key, pattern in patterns.items():
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            metrics[key] = matches[0]
    
    return metrics

def compare_metrics(readme_metrics: Dict, doc_metrics: Dict) -> List[Tuple[str, str, str, bool]]:
    """Compare metrics from README and documentation."""
    results = []
    
    all_keys = set(readme_metrics.keys()) | set(doc_metrics.keys())
    
    for key in sorted(all_keys):
        readme_val = readme_metrics.get(key, "NOT FOUND")
        doc_val = doc_metrics.get(key, "NOT FOUND")
        match = readme_val == doc_val
        results.append((key, readme_val, doc_val, match))
    
    return results

def main():
    """Main verification function."""
    print("[*] README Metrics Verification")
    print("=" * 80)
    print()
    
    # Paths
    readme_path = Path("README.md")
    doc_path = Path("docs/PROJECT_DOCUMENTATION.md")
    
    if not readme_path.exists():
        print(f"[X] README not found: {readme_path}")
        return
    
    if not doc_path.exists():
        print(f"[X] Documentation not found: {doc_path}")
        return
    
    # Read files
    print("[+] Reading files...")
    readme_content = readme_path.read_text(encoding='utf-8')
    doc_content = doc_path.read_text(encoding='utf-8')
    
    # Extract metrics
    print("[+] Extracting metrics...")
    readme_metrics = extract_metrics(readme_content, "README")
    doc_metrics = extract_metrics(doc_content, "PROJECT_DOCUMENTATION")
    
    print(f"   Found {len(readme_metrics)} metrics in README")
    print(f"   Found {len(doc_metrics)} metrics in PROJECT_DOCUMENTATION")
    print()
    
    # Compare
    print("[=] Comparison Results:")
    print("-" * 80)
    print(f"{'Metric':<40} {'README':<15} {'DOCS':<15} {'Match'}")
    print("-" * 80)
    
    results = compare_metrics(readme_metrics, doc_metrics)
    
    mismatches = []
    for key, readme_val, doc_val, match in results:
        status = "[OK]" if match else "[DIFF]"
        print(f"{key:<40} {readme_val:<15} {doc_val:<15} {status}")
        if not match:
            mismatches.append((key, readme_val, doc_val))
    
    print("-" * 80)
    print()
    
    # Summary
    total = len(results)
    matched = sum(1 for _, _, _, m in results if m)
    
    print(f"[*] Summary:")
    print(f"   Total metrics compared: {total}")
    print(f"   Matches: {matched} ({matched/total*100:.1f}%)")
    print(f"   Mismatches: {len(mismatches)} ({len(mismatches)/total*100:.1f}%)")
    print()
    
    if mismatches:
        print("[!] ACTION REQUIRED:")
        print("   The following metrics in README.md need to be updated to match")
        print("   PROJECT_DOCUMENTATION.md (the source of truth):")
        print()
        for key, readme_val, doc_val in mismatches:
            print(f"   - {key}:")
            print(f"     README has: {readme_val}")
            print(f"     SHOULD BE:  {doc_val}")
            print()
    else:
        print("[OK] All metrics match! README is in sync with documentation.")
    
    # Manual verification needed
    print()
    print("[!] MANUAL VERIFICATION NEEDED:")
    print("   This script does automated pattern matching. Please also manually verify:")
    print("   1. Confidence intervals (e.g., '0.848 AUC (95% CI: 0.812-0.883)')")
    print("   2. Sample sizes in tables")
    print("   3. Feature counts (e.g., '14 biomarkers', '21 features')")
    print("   4. All percentage values")
    print("   5. Dataset sizes (GB, MB)")
    
    return len(mismatches) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
