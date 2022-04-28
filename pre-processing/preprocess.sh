../mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < corpus.en > tokenized.en 
../mosesdecoder/scripts/tokenizer/tokenizer.perl -l de < corpus.de > tokenized.de 
../mosesdecoder/scripts/recaser/train-truecaser.perl --model corpus.en --corpus tokenized.en
../mosesdecoder/scripts/recaser/train-truecaser.perl --model corpus.de --corpus tokenized.de
../mosesdecoder/scripts/recaser/truecase.perl --model corpus.en < tokenized.en > truecased.en
../mosesdecoder/scripts/recaser/truecase.perl --model corpus.de < tokenized.de > truecased.de
subword-nmt get-vocab --input truecased.en --output vocab.en
subword-nmt get-vocab --input truecased.de --output vocab.de