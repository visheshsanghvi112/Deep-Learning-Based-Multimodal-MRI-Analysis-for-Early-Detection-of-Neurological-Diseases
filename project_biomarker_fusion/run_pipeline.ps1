# Run All Pipeline Steps
# =======================

Write-Host "=" -NoNewline -ForegroundColor Cyan; Write-Host "="*79 -ForegroundColor Cyan
Write-Host "BIOMARKER + MRI FUSION - COMPLETE PIPELINE" -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan; Write-Host "="*79 -ForegroundColor Cyan

Write-Host "`n⚠️  SAFETY CHECK:" -ForegroundColor Yellow
Write-Host "   This pipeline does NOT modify any existing work!" -ForegroundColor Yellow
Write-Host "   All results go to project_biomarker_fusion/ only`n" -ForegroundColor Yellow

$continue = Read-Host "Continue? (y/n)"
if ($continue -ne "y") {
    Write-Host "Aborted." -ForegroundColor Red
    exit
}

# Step 1: Extract biomarkers
Write-Host "`n[1/4] Extracting biomarkers from ADNIMERGE..." -ForegroundColor Green
python src\01_extract_biomarkers.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in step 1!" -ForegroundColor Red
    exit 1
}

# Step 2: Prepare fusion dataset
Write-Host "`n[2/4] Preparing fusion dataset..." -ForegroundColor Green
python src\02_prepare_fusion_data.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in step 2!" -ForegroundColor Red
    exit 1
}

# Step 3: Train fusion model
Write-Host "`n[3/4] Training multimodal fusion model..." -ForegroundColor Green
python src\03_train_fusion.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in step 3!" -ForegroundColor Red
    exit 1
}

# Step 4: Evaluate and compare
Write-Host "`n[4/4] Evaluating and comparing results..." -ForegroundColor Green
python src\04_evaluate.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in step 4!" -ForegroundColor Red
    exit 1
}

Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline -ForegroundColor Green; Write-Host "="*79 -ForegroundColor Green
Write-Host "✅ PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Green; Write-Host "="*79 -ForegroundColor Green

Write-Host "`nResults:" -ForegroundColor Cyan
Write-Host "  - results/metrics.json         : Model performance"
Write-Host "  - results/comparison.json      : Comparison with baselines"
Write-Host "  - results/model_comparison.png : Visualization"
Write-Host "  - results/checkpoints/         : Trained model weights"

Write-Host "`n📊 Check the results and update documentation!`n" -ForegroundColor Yellow
