python make_patch.py --train_input_img ../data/train_input_img --train_label_img ../data/train_label_img --to_save_folder ../data && python training_phase1.py --data_path /workspace/smart_lg_life_team/data && chmod 755 ./weights_train/model_unet_best.pt && python training_phase2.py --data_path /workspace/smart_lg_life_team/data