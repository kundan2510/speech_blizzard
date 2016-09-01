TRAIN_DATA_FOLDER=/Tmp/kumarkun/blizzard_flac
OUT_FOLDER=/Tmp/kumarkun/blizzard_pixelCNN
OTHER_INFO=GRU_2_pxCNN_2_wd_16_dep_32_bs_32
N_FILES=360000
THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32 python pixelCNN_GRU.py -br 16000 -tdata $TRAIN_DATA_FOLDER -n_files $N_FILES -o $OUT_FOLDER  -oi $OTHER_INFO -np 2 -n_gru 2 -wd 16 -dep 32 -bs 32


