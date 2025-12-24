"""
This function is used to print the tokenized text in a colorized way.
It is used to visualize the differences between the two tokenizers, including LZW tokenizer.

N.B.
- The colors may vary depending on the terminal and the OS, we used iTerm2 on MacOS.
- The display is not perfect on bash or zsh with iTerm2, but enable tmux session produces a perfect display.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from color_tokenizer import ColorLZW_Tokenizer, HuggingFaceTokenizer
from _legacy_lzw_tokenizer import Legacy_LZW_Tokenizer

# -------------------------
# Tokenizer Initialization
# -------------------------

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

hf_tokenizer = HuggingFaceTokenizer(MODEL_NAME)
lzw_tokenizer = ColorLZW_Tokenizer(
    Legacy_LZW_Tokenizer(
        AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True),
        disable_whitespace=True,
    )
)


# -------------------------
# Text Examples
# -------------------------

code_example = """
class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True

        self.pruned_heads = set()
    """

text_example = """
A quick brown fox jumps over the lazy dog.
A quick brown fox jumps over the lazy dog.
A quick brown fox jumps over the lazy dog.
A quick brown fox jumps over the lazy dog.
A quick brown fox jumps over the lazy dog.
A quick brown fox jumps over the lazy dog.
A quick brown fox jumps over the lazy dog.
A quick brown fox jumps over the lazy dog.
"""

biomedical_text = """
The brief existence of an mRNA molecule begins with transcription, and ultimately ends in degradation. During its life, an mRNA molecule may also be processed, edited, and transported prior to translation. Eukaryotic mRNA molecules often require extensive processing and transport, while prokaryotic mRNA molecules do not. A molecule of eukaryotic mRNA and the proteins surrounding it are together called a messenger RNP.[citation needed]

Transcription
Main article: Transcription (genetics)
Transcription is when RNA is copied from DNA. During transcription, RNA polymerase makes a copy of a gene from the DNA to mRNA as needed. This process differs slightly in eukaryotes and prokaryotes. One notable difference is that prokaryotic RNA polymerase associates with DNA-processing enzymes during transcription so that processing can proceed during transcription. Therefore, this causes the new mRNA strand to become double stranded by producing a complementary strand known as the tRNA strand, which when combined are unable to form structures from base-pairing. Moreover, the template for mRNA is the complementary strand of tRNA, which is identical in sequence to the anticodon sequence that the DNA binds to. The short-lived, unprocessed or partially processed product is termed precursor mRNA, or pre-mRNA; once completely processed, it is termed mature mRNA.[citation needed]

Uracil substitution for thymine
mRNA uses uracil (U) instead of thymine (T) in DNA. uracil (U) is the complementary base to adenine (A) during transcription instead of thymine (T). Thus, when using a template strand of DNA to build RNA, thymine is replaced with uracil. This substitution allows the mRNA to carry the appropriate genetic information from DNA to the ribosome for translation. Regarding the natural history, uracil came first then thymine; evidence suggests that RNA came before DNA in evolution.[1] The RNA World hypothesis proposes that life began with RNA molecules, before the emergence of DNA genomes and coded proteins. In DNA, the evolutionary substitution of thymine for uracil may have increased DNA stability and improved the efficiency of DNA replication.[2][3]

Eukaryotic pre-mRNA processing
Main article: Post-transcriptional modification

DNA gene is transcribed to pre-mRNA, which is then processed to form a mature mRNA, and then lastly translated by a ribosome to a protein
Processing of mRNA differs greatly among eukaryotes, bacteria, and archaea. Non-eukaryotic mRNA is, in essence, mature upon transcription and requires no processing, except in rare cases.[4] Eukaryotic pre-mRNA, however, requires several processing steps before its transport to the cytoplasm and its translation by the ribosome.

Splicing
Main article: RNA splicing
The extensive processing of eukaryotic pre-mRNA that leads to the mature mRNA is the RNA splicing, a mechanism by which introns or outrons (non-coding regions) are removed and exons (coding regions) are joined. [5][6]

5' cap addition
Main article: 5' cap

5' cap structure
A 5' cap (also termed an RNA cap, an RNA 7-methylguanosine cap, or an RNA m7G cap) is a modified guanine nucleotide that has been added to the "front" or 5' end of a eukaryotic messenger RNA shortly after the start of transcription. The 5' cap consists of a terminal 7-methylguanosine residue that is linked through a 5'-5'-triphosphate bond to the first transcribed nucleotide. Its presence is critical for recognition by the ribosome and protection from RNases.[citation needed]

Cap addition is coupled to transcription, and occurs co-transcriptionally, such that each influences the other. Shortly after the start of transcription, the 5' end of the mRNA being synthesized is bound by a cap-synthesizing complex associated with RNA polymerase. This enzymatic complex catalyzes the chemical reactions that are required for mRNA capping. Synthesis proceeds as a multi-step biochemical reaction.[citation needed]

Editing
In some instances, an mRNA will be edited, changing the nucleotide composition of that mRNA. An example in humans is the apolipoprotein B mRNA, which is edited in some tissues, but not others. The editing creates an early stop codon, which, upon translation, produces a shorter protein. Another well-defined example is A-to-I (adenosine to inosine) editing, which is carried out by double-strand specific adenosine-to inosine editing (ADAR) enzymes. This can occur in both the open reading frame and untranslated regions, altering the structural properties of the mRNA. Although essential for development, the exact role of this editing is not fully understood [7]

Polyadenylation
Main article: Polyadenylation

Polyadenylation
Polyadenylation is the covalent linkage of a polyadenylyl moiety to a messenger RNA molecule. In eukaryotic organisms most messenger RNA (mRNA) molecules are polyadenylated at the 3' end, but recent studies have shown that short stretches of uridine (oligouridylation) are also common.[8] The poly(A) tail and the protein bound to it aid in protecting mRNA from degradation by exonucleases. Polyadenylation is also important for transcription termination, export of the mRNA from the nucleus, and translation. mRNA can also be polyadenylated in prokaryotic organisms, where poly(A) tails act to facilitate, rather than impede, exonucleolytic degradation.[citation needed]

Polyadenylation occurs during and/or immediately after transcription of DNA into RNA. After transcription has been terminated, the mRNA chain is cleaved through the action of an endonuclease complex associated with RNA polymerase. After the mRNA has been cleaved, around 250 adenosine residues are added to the free 3' end at the cleavage site. This reaction is catalyzed by polyadenylate polymerase. Just as in alternative splicing, there can be more than one polyadenylation variant of an mRNA.

Polyadenylation site mutations also occur. The primary RNA transcript of a gene is cleaved at the poly-A addition site, and 100–200 A's are added to the 3' end of the RNA. If this site is altered, an abnormally long and unstable mRNA construct will be formed.

Transport
Another difference between eukaryotes and prokaryotes is mRNA transport. Because eukaryotic transcription and translation is compartmentally separated, eukaryotic mRNAs must be exported from the nucleus to the cytoplasm—a process that may be regulated by different signaling pathways.[9] Mature mRNAs are recognized by their processed modifications and then exported through the nuclear pore by binding to the cap-binding proteins CBP20 and CBP80,[10] as well as the transcription/export complex (TREX).[11][12] Multiple mRNA export pathways have been identified in eukaryotes.[13]

In spatially complex cells, some mRNAs are transported to particular subcellular destinations. In mature neurons, certain mRNA are transported from the soma to dendrites. One site of mRNA translation is at polyribosomes selectively localized beneath synapses.[14] The mRNA for Arc/Arg3.1 is induced by synaptic activity and localizes selectively near active synapses based on signals generated by NMDA receptors.[15] Other mRNAs also move into dendrites in response to external stimuli, such as β-actin mRNA.[16] For export from the nucleus, actin mRNA associates with ZBP1[17] and later with 40S subunit. The complex is bound by a motor protein and is transported to the target location (neurite extension) along the cytoskeleton. Eventually ZBP1 is phosphorylated by Src in order for translation to be initiated.[18] In developing neurons, mRNAs are also transported into growing axons and especially growth cones. Many mRNAs are marked with so-called "zip codes", which target their transport to a specific location.[19][20] mRNAs can also transfer between mammalian cells through structures called tunneling nanotubes.[21][22]

Translation
Main article: Translation (biology)

Translation of mRNA to protein
Because prokaryotic mRNA does not need to be processed or transported, translation by the ribosome can begin immediately after the end of transcription. Therefore, it can be said that prokaryotic translation is coupled to transcription and occurs co-transcriptionally. [23]

Eukaryotic mRNA that has been processed and transported to the cytoplasm (i.e., mature mRNA) can then be translated by the ribosome. Translation may occur at ribosomes free-floating in the cytoplasm, or directed to the endoplasmic reticulum by the signal recognition particle. Therefore, unlike in prokaryotes, eukaryotic translation is not directly coupled to transcription. It is even possible in some contexts that reduced mRNA levels are accompanied by increased protein levels, as has been observed for mRNA/protein levels of EEF1A1 in breast cancer"""

chinese_text = (
    """
人工神经网络（英语：artificial neural network，ANNs）又称类神经网络，简称神经网络（neural network，NNs），在机器学习和认知科学领域，是一种模仿生物神经网络（动物的中枢神经系统，特别是大脑）的结构和功能的数学模型或计算模型，用于对函数进行估计或近似。神经网络由大量的人工神经元联结进行计算。大多数情况下人工神经网络能在外界信息的基础上改变内部结构，是一种自适应系统(adaptive system)，通俗地讲就是具备学习功能。现代神经网络是一种非线性统计性数据建模(概率模型)工具，神经网络通常是通过一个基于数学统计学类型的学习方法（learning method）得以优化，所以也是数学统计学方法的一种实际应用，通过统计学的标准数学方法我们能够得到大量的可以用函数来表达的局部结构空间，另一方面在人工智能学的人工感知领域，我们通过数学统计学的应用可以来做人工感知方面的决定问题（也就是说通过统计学的方法，人工神经网络能够类似人一样具有简单的决定能力和简单的判断能力），这种方法比起正式的逻辑学推理演算更具有优势。

和其他机器学习方法一样，神经网络已经被用于解决各种各样的问题，例如机器视觉和语音识别。这些问题都是很难被传统基于规则的编程所解决的。"""
    * 3
)

french_text = """
L'impressionnisme est un mouvement pictural apparu en France dans les années 1860 en opposition à l'art académique et visant à représenter le caractère éphémère de la lumière et ses effets sur les couleurs et les formes. Le groupe des impressionnistes se forme autour d'Édouard Manet, chef de file de l'avant-garde artistique dans les années 1860, qui ne participe cependant à aucune exposition impressionniste. Après plusieurs scandales et refus au Salon, la grande exposition annuelle d'artistes agréés par l'Académie des Beaux-Arts, de jeunes artistes décident de s'associer pour organiser des expositions indépendantes. Cette idée se concrétise en 1874, dans une exposition qui réunit trente artistes dont Paul Cézanne, Edgar Degas, Claude Monet, Berthe Morisot, Camille Pissarro, Auguste Renoir et Alfred Sisley. Le journaliste satirique Louis Leroy invente alors le terme « impressionnisme » à partir du tableau Impression, soleil levant de Monet, devenu depuis le nom du mouvement. Les artistes subissent d'abord des critiques violentes de la part de la presse et du public, mais ils sont soutenus par des collectionneurs qui permettent la tenue de leurs premières expositions, notamment Gustave Caillebotte.

L'impressionnisme commence à être accepté en 1880, grâce au soutien du nouveau gouvernement de Léon Gambetta et de critiques comme Émile Zola. Les œuvres font petit à petit leur entrée dans les musées, au Salon des artistes français, qui succède au Salon de l'Académie des Beaux-Arts, et sur le marché de l'art. Le marchand Paul Durand-Ruel joue un rôle crucial dans le soutien et la diffusion de l'impressionnisme, qui s'exporte aux États-Unis à partir de 1886, grâce à la peintre Mary Cassatt. Le mouvement y obtient un grand succès, qui participe à la consécration de Monet et au développement d'écoles impressionnistes hors de France au cours des années 1890. Cette décennie voit la mort de Morisot, Caillebotte et Sisley et la dispersion du groupe, tandis que se développent de nouvelles avant-gardes auxquelles adhèrent certains impressionnistes, comme Cézanne et Pissarro.

Les artistes impressionnistes créent une nouvelle esthétique opposée à l'art académique. Leur style apparaît pour la première fois dans les toiles peintes par Monet et Renoir à l'île de la Grenouillère, en 1869. Ils font primer la couleur sur le dessin, utilisent des compositions inhabituelles et une touche rapide, et composent généralement en plein air sur le motif. Tournés vers des sujets modernes, ils représentent principalement des paysages, des scènes de la vie intime et les loisirs de leur époque.

Les impressionnistes dans les collections muséales
L'impressionnisme étant un mouvement largement porté par des artistes français, il est logique de trouver de nombreuses œuvres dans des musées situés en France ; toutefois, la plupart des grandes collections d'art moderne à travers le monde s'efforcent également de présenter au moins quelques exemples de toiles impressionnistes.

Ainsi, le musée Getty à Los Angeles présente de nombreuses œuvres typiques du mouvement comme Soleil levant (marine) de Monet qui constitue un proche cousin du célèbre Impression, soleil levant du musée Marmottan[158]. D'autres grands musées américains abritent des collections impressionnistes importantes comme le Metropolitan Museum of Art de New York[159], le musée des Beaux-Arts de Boston avec notamment pas moins de 35 tableaux de Monet[160] ou encore l'Art Institute of Chicago[161] ; en Russie, le musée de l'Ermitage à Saint-Pétersbourg présente de nombreuses œuvres de Monet ou encore de Renoir[162], de même que le musée national de l'Art occidental de Tokyo à partir de la collection Matsukata[163], etc.

En France, la plus grande concentration d'œuvres impressionnistes se trouve au musée d'Orsay à Paris[164], qui a d'ailleurs organisé en avril 2024 une exposition à l'occasion des 150 ans de la première édition de 1874[165]. Mais d'autres établissements accueillent un fonds important, comme le musée Marmottan Monet (avec une vaste collection de toiles de Monet bien sûr, mais aussi de Berthe Morisot parmi beaucoup d'autres)[166] ou bien le musée des Beaux-Arts de Rouen[167]. À Giverny, ville où vécut et travailla Claude Monet, le musée local prend en 2009 le nom de « musée des impressionnismes »[168].
"""
biomedical_output = """
<|user|> Please explain what is messenger ribonucleic acid<|end|><|assistant|> Messenger RNA (mRNA) is a type of ribonucleic acid (RNA) that serves as the messenger carrying genetic information from DNA to the ribosomes, where the protein synthesis takes place. The genetic code on the DNA sequence is transcribed into mRNA in a process called transcription.

As it moves out of the nucleus of the cell, the mRNA strand encounters the ribosomes that read the sequence and use it as a guide to create a protein in a process known as translation. mRNA has a sequence of nucleic acids according to which the cell makes the enzymes, vitamins, and many other enzymes needed for cellular function. Therefore, it is vital in all biological functions since proteins are required by the cell.

To summarize, mRNA is transcribed from DNA and is a type of RNA that serves as a messenger to carry genetic information within the cell. It instructs ribosomes on how to build particular proteins and, in turn, carries out the process of protein synthesis.<|end|><|user|> Can you give me some examples of how to identify mRNA in laboratory experiments?<|end|><|assistant|> Sure! Here are some examples of how to identify mRNA in laboratory experiments:

1. Northern Blotting (RNA-Protein Blots): This technique is based on separating RNA molecules according to their size on the basis of polyacrylamide gel

"""

french_output = """
<|user|> Expliquez-moi l'histoire de la Tour Eiffel<|end|><|assistant|> La Tour Eiffel est une tour à étages émaillée située à Paris, France. Elle a été conçue par l'ingénieur Gustave Eiffel, dont le nom est associé à la construction de cette tour. La construction de la tour a été achevée en 1889 pour célébrer le 100e anniversaire de la Révolution française.

La Tour Eiffel, haute de 324 mètres, est devenue le symbole de Paris et son point de repère le plus célèbre dans le monde. Elle est également l'un des monuments les plus visités au monde avec plus de 7 millions de visiteurs chaque année. La tour est utilisée comme balcon panoramique offrant des vues magnifiques sur Paris et la région.

Malgré les critiques initiales de nombreux Parisiens sur la conception proéminente de cette tour, elle reste le plus populaire du monde. En effet, la Tour Eiffel est la seule structure du type pyramide avant l'escalier mobile de visint ses côtés.

La Tour Eiffel a subi de nombreux changements et modifications au cours de ses 133 ans de vie. En 1981, elle a reçu la distinction "Monument Historique" par le ministre de la Cultura France. Et après la Seconde Guerre Mondiale, la tour est restée la carte par l'attrait touristique de la ville de Parisiens. Aujourd'hui, la tour intéresse pas moins de gourmands, entre autres artistes du monde, qui viennent admirer les vues touristique et son...vintageégo!<|end|>"""

python_code_output = """
 Implement a Transformer model in PyTorch that takes a sequence of tokens and outputs a hidden representation for each token. The Transformer should consist of an embedding layer, a self-attention mechanism, and a feed-forward neural network. Ensure that the Transformer can handle variable input and output sizes, as well as different sequence lengths.<|end|><|assistant|> Here is an example implementation of a Transformer model in PyTorch:

```
import torch
from torch.nn import Linear, LayerNorm, Module, Unfold, ModuleList

class TransformerLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_attn = nf.dropout
        self.attn = MultiheadAttn(d_model, nhead, dim_feedforward, dropout=self.dropout_attn)
        self.layernorm = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dim_feedforward) # TODO - implement the feedforward neural network

    def __call__(self, x, dropout=0):
        # Apply an embedding layer to the input sequence

"""


# -------------------------
# Visualization
# -------------------------


def visualize_text(text):
    print(f"\n--- (LZW Tokenizer) ---")
    lzw_tokenizer.print_pretty(text)

    print(f"\n--- (HuggingFace Tokenizer) ---")
    hf_tokenizer.print_pretty(text)


if __name__ == "__main__":
    visualize_text(code_example)
