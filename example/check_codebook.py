text = """
The École Polytechnique Fédérale de Lausanne (French pronunciation: [ekɔl pɔlitɛknik fedeʁal də lɔzan], EPFL) is a public research university in Lausanne, Switzerland, founded in 1969 with the mission to "train talented engineers in Switzerland".

Like its sister institution ETH Zurich, EPFL is part of the Swiss Federal Institutes of Technology Domain[7] which groups several universities and research institutes under the Federal Department of Economic Affairs, Education and Research.[8] As of 2024, EPFL enrolled 14,012 students from over 130 countries.

EPFL has an urban campus that extends alongside Lake Geneva, and includes the EPFL Innovation Park as well as university research centers and affiliated laboratories.
"""


import os, sys
from transformers import AutoTokenizer, PreTrainedTokenizerBase

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from _legacy_lzw_tokenizer import Legacy_LZW_Tokenizer

phi_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", use_fast=True
)
lzw_tokenizer = Legacy_LZW_Tokenizer(phi_tokenizer)


lzw_tokenization = lzw_tokenizer.encode(text)
print(lzw_tokenization.token_ids)
print(lzw_tokenization.codebook.pad())
