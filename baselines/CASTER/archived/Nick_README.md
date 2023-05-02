
This is Nick's README for using CASTER's codebase.

Just to summarize the process of CASTER, CASTER has an SPM (Sequential Pattern Mining Algorithm) module that encodes the SMILEs compound sinto their one-hot representation. Then the one-hots between the two compounds are OR-ed together (1 if substructure is present in one of the two compounds, 0 otherwise), then the Encoder + Dictionary modules encode/decode to make the binary prediction.

The Encoding/Decoding setup is in the pretraining such as be able to train on larger "unlabeled" datasets first. A pretrained model parameters are saved. 

The predictor itself is a multilayer perceptron that outputs a binary classification. The default training setup that CASTER uses is fixed negative samples. 

From my investigation of the CASTER Codebase, they supply the most common substructures in "subword_units_map.csv" however they do not supply the code itself for mining the substructures.

The input for the fine-tuning step takes in the following format:
- drug_index, Drug1_SMILES, drug_index, Drug2_SMILES,label

preprocessing.ipynb is my attempt to take our PrimeKG data and convert it into the input that can be taken in by CASTER. 

run_caster-our-split.ipynb is my attempt to run caster fine-tuned on our dataset. 











