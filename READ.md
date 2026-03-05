MarketMuse: A play on "marketing muse," suggesting it's a source of inspiration for marketers.

first decided the model to be fine tuned 
casual or seq-2-seq

preprocess data in the formm model accepts 
for no input use alpaca format bu here input was requred so does not strictly follow that
seperate the role as user and system and extract content of both 
 use autotokinizer for tokinization 
 the traget and input will be tokenized seperately 
 now for tokenization have 2 options pad or eos (used for seq-2-seq)
 first check if model alreday use it or not than add
 masking behaviour 
 encoder, decoder,seq-2-seq