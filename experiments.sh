TRAIN_DATA=/Tmp/kumarkun/blizzard_flac
OUT_FOLDER=/Tmp/kumarkun/blizzard_pixelCNN
OTHER_INFO=GRU_1_pxCNN_2_wd_32_dep_32_bs_64

THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python pixelCNN_GRU.py -br 16000 -tdata $TRAIN_DATA_FOLDER -o $OUT_FOLDER  -oi $OTHER_INFO -n_gru 1 -wd 32 -dep 32 -bs 64


