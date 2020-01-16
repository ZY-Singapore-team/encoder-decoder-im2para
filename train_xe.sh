python -W ignore train.py \
    --batch_size 16 \
    --input_att_dir '/data/liangming/parabu_att' \
    --input_fc_dir '/data/liangming/parabu_fc' \
    --input_json 'data/paratalk/paratalk.json' \
    --input_label_h5 'data/paratalk/paratalk_label.h5' \
    --language_eval 1 \
    --learning_rate 5e-4 \
    --learning_rate_decay_start -1 \
    --scheduled_sampling_start -1 \
    --max_epochs 30 \
    --rnn_type 'lstm' \
    --input_encoding_size 768 \
    --val_images_use 5000 \
    --save_checkpoint_every 500 \
    --checkpoint_path 'bert_test2/' \
    --id 'bert2' \
    --print_freq 20 \
    --beam_size 1
    # --start_from 'bert_test2/'
