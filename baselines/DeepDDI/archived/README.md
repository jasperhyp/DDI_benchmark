This is my documentation.

I am goign to create a new preprocessing script that doesn't just for loop over everything. 

NOTE: Any documents that have the "FEB" extension are outdated. 

Deep_Embeds_DrugCentricSplit_TWOSIDES.ipynb [Current]
- This is the python notebook that tries to both:
    1. Preprocess the chemical compounds for Morgan FP or Tanimoto
    2. Create the Lookup table for PCA fit similarity scores

    3. Taking the TWOSIDEs drug splits, create the appropriate multilabel preprocessing for
        the multilabel classification. I struggled to get this to work in a way that is efficient enough to run multiple times and doesn't explode due to memory.


Deep_GetLookUp_Embeds_Feb.ipynb [Outdated]
- This is a python notebook that:
    1. Preprocess the chemical compounds for Morgan FP or Tanimoto
    2. Create the Lookup table for PCA fit similarity scores
    - outputs the lookup table
- This file is outdated due to the unique TWOSIDEs splits that Yepeng prepared


Deep_DrugCentricSplit_Feb.ipynb [Outdated]
- This python notebook takes the outputs of Deep_GetLookUp_Embeds_Feb.ipynb to create the actual       files that can be used to train DeepDDI


Deep_PreProcess_Collated_Feb.py [Outdated]
- This is the (outdated) python file that is effectively both Deep_GetLookUp_Embeds_Feb.ipynb and Deep_DrugCentricSplit_Feb.ipynb in one script. but this file is even more outdated. I am keeping this here just for reference, but feel free to delete. 



### Running the Training + Testing Scripts

train_Twosides.py [Current] but incomplete
train_original.py [Outdated] but kept for reference
- This is the DeepDDI training that was set up based on the original PrimeKG binary data splits
    the most notable components of this is that it contains 

Helper files:
- utils.py [Current]
- metrics.py [Current]
- parse_args.py [Current]



