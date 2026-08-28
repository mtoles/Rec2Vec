./.venv/bin/python rephrase_dataset.py \
  dataset/processed/feature-distance-dataset_gemini-2.5-flash_1000000_nolek \
  --workers 4 --max-workers 256


./.venv/bin/python rephrase_dataset.py \
  dataset/processed/deepfashion-inshop-image-triplets_hf_20000 \
  --descriptions dataset/deepfashion-inshop-image-triplets_hf_20000.jsonl \
  --workers 4 --max-workers 256