

## Start EmbedDTI

### Introduction

The identification of drug-target interaction (DTI) plays a key role in drug discovery and development. Benefitting from large-scale drug databases and verified DTI relationships, a lot of machine learning methods have been developed to predict DTIs. However, due to the difficulty in extracting useful information from molecules, the performance of these methods is limited by the representation of drugs and target proteins. 
This study proposes a new model called EmbedDTI to enhance the representation for both drugs and target proteins, and improve the performance of DTI prediction. For protein sequences, we leverage language modeling for pre-training the feature embeddings of amino acids and feed them to a CNN model for further representation learning. For drugs, we build two levels of graphs to represent compound structural information, namely the atom graph and substructure graph, and adopt GCN with an attention module to learn the embedding vectors for the graphs. We compare EmbedDTI with the existing DTI predictors on two benchmark datasets. The experimental results show that EmbedDTI outperforms the state-of-the-art models by large margins and the attention module can identify the components crucial for DTIs in compounds.

### Models

We provide the following models:

* EmbedDTI_ori:  i.e, EmbedDTI_noPre mentioned in EmbedDTI paper, we represent the original protein sequence in one-hot vector and represent the input vector as a random embedding vector through an embedding layer to input it into the CNN module. For drugs, we convert their SMILES sequences into two graph structures (atom and substructure) to retain as much structural information as possible for feature learning. 
* EmbedDTI_pre: i.e, EmbedDTI_noAttn mentioned in EmbedDTI paper. For protein sequences, we use the GloVe algorithm to obtain the pre-trained embedding representations of amino acids. The drug part is the same as EmbedDTI_ori.
* EmbedDTI: Based on EmbedDTI_pre model, we add a scaled dot-product attention layer before the GCN network for atom and substructure branch to help learn the relative importance of each node (atom or substructure).

### Step-by-step running

1. Please install pytorch-geometric and rdkit software.


2. If you just want to predict the affinity scores from the pretrained models, you can use the Linux command:

   `python3 predict_with_davis_orimodel.py 0` for predict EmbedDTI_ori on Davis dataset; The result is saves as "results/result_davis_ori.csv";

   `python3 predict_with_davis_premodel.py 0` for predict EmbedDTI_pre on Davis dataset; The result is saves as "results/result_davis_pre.csv";

   `python3 predict_with_davis_attnmodel.py 0` for predict EmbedDTI on Davis dataset; The result is saves as "results/result_davis_attn.csv";

   `python3 predict_with_kiba_orimodel.py 0` for predict EmbedDTI_ori on KIBA dataset; The result is saves as "results/result_kiba_ori.csv";

   `python3 predict_with_kiba_premodel.py 0` for predict EmbedDTI_pre on KIBA dataset; The result is saves as "results/result_kiba_pre.csv";

   `python3 predict_with_kiba_attnmodel.py 0` for predict EmbedDTI on KIBA dataset; The result is saves as "results/result_kiba_attn.csv";

3. Create protein sequences and drug componds into pytorch format:

   `python3 dataloader_davis.py` for create data in Davis dataset;

   `python3 dataloader_kiba.py` for create data in KIBA dataset;

4. Train a EmbedDTI prediction model:

   `python3 training_davis.py 0 0`  to train a model using Davis dataset. The first argument is for the index of the models, where 0 indicates Embed_ori, 1 indicates EmbedDTI_pre and 2 indicates EmbedDTI. The second argument is for the index of the cuda.  We save the results of each epoch during the training process.

   `python3 training_kiba.py 0 0`  to train a model using KIBA dataset. The argument settings are the same as on the Davis dataset. We save the results of each epoch during the training process.

