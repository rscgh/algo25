import os
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from glob import glob
from tqdm.auto import tqdm
from brainannlib.utils import get_root_dir, get_output_actvations_dir, get_hf_dir

root_data_dir = get_root_dir()
actv_dir = get_output_actvations_dir()

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# print some information on the GPU
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.get_device_properties(torch.cuda.current_device()))

# Custom dataset to hold the sentences
from brainannlib.stimuli_data_loading import SentenceDataset

# Custom collate function to handle tokenization
def collate_fn(batch):
    encoding = tokenizer.batch_encode_plus(batch, padding=True, return_tensors="pt", return_attention_mask=True)
    return encoding['input_ids'], encoding['attention_mask']


#all_tsv_files = glob("/scratch/users/robert.scholz2/cneuromod/algonauts_2025.competitors/stimuli/transcripts/**/**/*.tsv")
all_tsv_files = glob(f"{root_data_dir}/algonauts_2025.competitors/stimuli/transcripts/**/**/*.tsv")
file = all_tsv_files[0]
print(len(all_tsv_files), "e.g.", file)

# prepare all
kept_tokens_last_hidden_state = 6
device = "cuda" # "cpu"
#checkpoint = "HuggingFaceTB/SmolLM2-135M"
checkpoint = "HuggingFaceTB/SmolLM2-1.7B"

# for fp16 use `torch_dtype=torch.float16`
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# set the padding id and token to be able to create padded batches
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

# load the LLM itself
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16).eval()

#Optional testing of th emodel
#inputs = tokenizer.encode("Gravity is", return_tensors="pt").to(device)
#outputs = model.generate(inputs)
#print(tokenizer.decode(outputs[0]))

import os
# iterate over all the available movie/episode transcript files 
# and collect the activations
for j, transcript_file in tqdm(enumerate(all_tsv_files), total=len(all_tsv_files)):
  
  print(f"{j+1}/{len(all_tsv_files)}", id, end=" ")

  #transcript_file e.g. "../../friends_s01e01a.tsv" -> id: friends_s01e01a
  id = transcript_file.split("/")[-1].split(".")[0]

  # if the output activation file already exisis, skip the current iteration
  fn = f"{actv_dir}actv-SmolLM2-1.7B-{id}-last500wodslast5trs.npy"
  if os.path.exists(fn): continue; 

  # read in the tsv file & replace nans
  df = pd.read_csv(transcript_file, sep = '\t')
  df.insert(loc=0, column="is_na", value=df["text_per_tr"].isna())
  # list of the text during each TR
  text_per_tr = df["text_per_tr"].replace(np.nan, "").tolist()

  # create a pytorch dataset & dataloader for efficient loading+preparation of batches
  dataset = SentenceDataset(text_per_tr, mode="n_used_words", n_used_words=500)
  dataloader = DataLoader(dataset, batch_size=40, shuffle=False, collate_fn=collate_fn)
  
  # empty array that will store all the model activations for each batch
  embd_data = []
  for input_ids, attention_mask in tqdm(dataloader, total=len(dataloader)):
    dev_inp = input_ids.to(device)
    with torch.no_grad():
      outputs = model(dev_inp, attention_mask=attention_mask.to(device), output_hidden_states=True)
    del dev_inp # freeing memory

    ## extraction of activations/embeddings/hidden-states of interest
    xhs = torch.stack(outputs.hidden_states).detach().cpu()
    for hs in outputs.hidden_states: del hs # freeing memory

    # find the index of the last input text token for each sample before padding
    last_token_indices = attention_mask.sum(dim=1) - 1
    # find the indices for the last x tokens before padding
    indices = np.array([np.clip(list(range(last_id - kept_tokens_last_hidden_state+1, last_id + 1)), a_min=0, a_max=None) for last_id in last_token_indices])
    
    # only keep the activations for a few layers to keep filesize manageable
    # also usually transformations within LLMs happen slowly, so equally spaced
    # probing should cover quite a bit of the information content.
    layer_idxs = np.linspace(0, len(xhs)-1, 5).round().astype(int)
    
    # subset the full collected activations/hidden states with the 
    # information compiled aboe
    embds = xhs[:, torch.arange(xhs.shape[1]).unsqueeze(1), indices]
    embds = embds[layer_idxs]

    torch.cuda.empty_cache();# freeing memory
    # appending the embeddings for the current batch
    embd_data.append(embds.to(torch.float16).numpy())

  # concatenating across all batches, then saving
  c=np.concatenate(embd_data, axis=1)
  print(c.shape)
  np.save(fn, c)
