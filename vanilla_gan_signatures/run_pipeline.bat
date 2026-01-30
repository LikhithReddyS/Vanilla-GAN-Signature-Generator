@echo off
echo ===================================================
echo Vanilla GAN Signature Generation Pipeline
echo ===================================================

REM 1. Preprocessing
echo [1/5] Preprocessing Data...
python -m src.preprocess_signatures --input_dir "D:\python\GAN SKILL PROJECT-2\signatures\full_org" --output_dir "data/processed"
if %errorlevel% neq 0 exit /b %errorlevel%

REM 2. Training
echo [2/5] Training GAN (1000 epochs)...
python -m src.train_vanilla_gan_signatures --epochs 1000 --batch_size 64 --data_dir "data/processed"
if %errorlevel% neq 0 exit /b %errorlevel%

REM 3. Generation
echo [3/5] Generating Synthetic Signatures...
python -m src.generate_signatures --checkpoint_path checkpoints/generator_final.pth --num_images 50 --output_dir data/synthetic
if %errorlevel% neq 0 exit /b %errorlevel%

REM 4. Verification Training
echo [4/5] Training Verification System (Augmented)...
python -m src.signature_verifier_train --real_data_dir "data/processed" --synthetic_data_dir "data/synthetic" --use_augmentation --epochs 10
if %errorlevel% neq 0 exit /b %errorlevel%

REM 5. App
echo [5/5] Launching UI...
echo Run 'python -m streamlit run src/app_vanilla_gan_signatures.py --server.fileWatcherType none' to launch the app manually.
python -m streamlit run src/app_vanilla_gan_signatures.py --server.fileWatcherType none

echo Done.
pause
