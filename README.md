# tampered-text-detection-in-document-image

Get TFR dataset
pip install -U huggingface_hub hf_xet
hf auth login
python download_tfr.py

cd ./datasets/TFR
& "$env:ProgramFiles\7-Zip\7z.exe" x .\TFR.zip
& "$env:ProgramFiles\7-Zip\7z.exe" x .\TFR -o".\TFR_extracted"