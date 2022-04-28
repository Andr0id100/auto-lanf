~/marian/build/marian \
 --model model.npz \
 --type transformer \
 --train-sets ~/lm-translation/news-commentary/truecased.en ~/lm-translation/news-commentary/truecased.de \
 --vocabs ~/lm-translation/news-commentary/vocab.en ~/lm-translation/news-commentary/vocab.de \
 --early-stopping 10 \
 --enc-depth 6 \
 --dec-depth 6 \
 --transformer-heads 8 \
 --tied-embeddings \
 --learn-rate 0.0003 \
 --lr-warmup 16000 \
 --lr-decay-inv-sqrt 16000 \
 --optimizer-params 0.9 0.98 1e-09 \
 --devices 0 1 \

