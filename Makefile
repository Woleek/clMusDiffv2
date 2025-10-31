encode_images:
	uv run scripts/encode_images.py --dataset_name_or_path Woleek/Img2Spec --output_file data/encodings_clip.p --batch_size 64

train_unet:
	uv run scripts/train_unet.py --encodings data/encodings_clip.p --deduplicate True --prediction_type v_prediction --pretrained_model models/clMusDiffv2_1 --start_epoch 46 --output_dir models/clMusDiffv2_2