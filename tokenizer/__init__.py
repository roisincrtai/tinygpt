"""Stage 3: the byte-level BPE tokenizer every stage shares."""
#
# `run` is NOT imported here. `python -m tokenizer.run` makes runpy import this package
# first, and if the module it is about to execute is already in sys.modules it runs a
# SECOND time as __main__ -- two module objects, two copies of every module-level value,
# and a RuntimeWarning. Import it explicitly where it is needed:
#
#     from tokenizer import run as train_bpe
from .bpe import BPETokenizer
