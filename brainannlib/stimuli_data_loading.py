######################################################################################
## This file provides functions that enable the efficient loading of stimuli datasets
## that then form the basis for the (fast) batched extraction of ANN layer activations 
##
## for algonauts the most relevant dataset definitons are:
##  TRSamplingDecordVDataset - for vision (from mkv-movie files)
##  SentenceDataset - for language (from tsv-subtitle files)
##  DecordAudioDataset - for audio (from mkv-movie files)
######################################################################################


##############################################################################
# Pytorch Dataloader and Datasets
##############################################################################

def get_pytorch_stim_dataset(dataset, stim_name, modality = "default", chunk_size_s=2):
    from lib.config import dataset_stimuli

    embd_mode = ""
    
    if dataset.startswith("NSD"):
        ds = NSD_DataSet(subset=stim_name) 

    elif dataset=="hcpmovies":
        stim_path = dataset_stimuli[dataset][stim_name]
        #target_mri_TR = 1.49 #s
        target_mri_TR = 1.0 #s
        if (modality == "default") or (modality=="vision"):
            ds = TRSamplingDecordVDataset(stim_path, target_mri_TR, None)
            #embd_mode = "eqsTR1.49s-"
            embd_mode = "eqsTR1s-"
            
        elif modality == "audio":
            ds = DecordAudioDataset(stim_path, transform=None, step_size_s=target_mri_TR, chunk_size_s=chunk_size_s)
            embd_mode = f"{chunk_size_s}sCNKperTR{target_mri_TR}s-";

    elif dataset.startswith("narratives"):
        stim_path = dataset_stimuli[dataset][stim_name]
        target_mri_TR=1.5;
        ds = WavAudioDataset(stim_path, step_size_s=target_mri_TR, chunk_size_s=chunk_size_s)
        embd_mode = f"{chunk_size_s}sCNKperTR{target_mri_TR}s-";

    elif dataset.startswith("CNM"):
        stim_path = dataset_stimuli[dataset][stim_name]
        target_mri_TR = 1.49 #s
        if (modality == "default") or (modality=="vision"):
            ds = TRSamplingDecordVDataset(stim_path, target_mri_TR, None)
            embd_mode = "eqsTR1.49s-"
            
        elif modality == "audio":
            ds = DecordAudioDataset(stim_path, transform=None, step_size_s=target_mri_TR, chunk_size_s=chunk_size_s)
            embd_mode = f"{chunk_size_s}sCNKperTR{target_mri_TR}s-";

    else:
        raise Exception("The loading for the dataset doesnt seem to be implemented")
        
    return ds, embd_mode


##############################################################################
# Video Datasets

import decord
decord.bridge.set_bridge('torch')
import numpy as np
from decord import VideoReader, AudioReader, cpu, AVReader
decord.bridge.set_bridge('torch')

from torch.utils.data.dataset import Dataset
import math 

class TRSamplingDecordVDataset(Dataset):
    def __init__(self, movie_file_path, target_mri_TR, transform=None, num_threads=1):
        self.videoreader = VideoReader(movie_file_path, num_threads=num_threads, ctx=cpu(0))
        self.videoreader.seek(0)
        self.transform = transform
        self.n_frames = len(self.videoreader)
        self.fps = self.videoreader.get_avg_fps();
        #self.n_tr_samples = math.ceil((self.n_frames/self.fps)/target_mri_TR)
        self.n_tr_samples = int(round((self.n_frames/self.fps)/target_mri_TR))

        tr_middle_s = (np.arange(self.n_tr_samples)*target_mri_TR)+target_mri_TR*0.5
        self.frame_idxs = np.round(tr_middle_s*self.fps).astype(int)
        # make sure it stays in the bounds
        self.frame_idxs = np.clip(self.frame_idxs, 0, self.n_frames-1)#.astype(int)
        
    def __len__(self): return len(self.frame_idxs)
    
    def __getitem__(self, idx):   
        mvidx = self.frame_idxs[idx]
        frame = self.videoreader[mvidx]
        if self.transform is not None:
            frame = self.transform(frame)
        return frame, 0
    

class TRSamplingDecordVDatasetV2(Dataset):
    def __init__(self, movie_file_path, target_mri_TR, imgs_per_tr=1, transform=None, num_threads=1, v=False):
        self.videoreader = VideoReader(movie_file_path, num_threads=num_threads, ctx=cpu(0))
        self.videoreader.seek(0)
        self.transform = transform
        self.n_frames = len(self.videoreader)
        self.fps = self.videoreader.get_avg_fps();
        #self.n_tr_samples = math.ceil((self.n_frames/self.fps)/target_mri_TR)
        self.n_trs = int(round((self.n_frames/self.fps)/target_mri_TR))

        tr_start_s = (np.arange(self.n_trs)*target_mri_TR)

        if imgs_per_tr == 1:
            # take the middle of the tr
            idxs =tr_start_s+target_mri_TR*0.5
        else:
            # take imgs_per_tr images tthrought the TR
            idxs = []
            offsets_s = np.linspace(0, target_mri_TR, imgs_per_tr+1)
            for j, start_s in enumerate(tr_start_s):
                if v and j <4: print((offsets_s+start_s)[1:])
                idxs = idxs+ list((offsets_s+start_s)[1:])
                # skip the first, as it will be equal to the last image of the prev TR

        self.frame_idxs = np.round(np.array(idxs)*self.fps).astype(int)
        # make sure it stays in the bounds
        self.frame_idxs = np.clip(self.frame_idxs, 0, self.n_frames-1)#.astype(int)
        
    def __len__(self): return len(self.frame_idxs)
    
    def __getitem__(self, idx):   
        mvidx = self.frame_idxs[idx]
        frame = self.videoreader[mvidx]
        if self.transform is not None:
            frame = self.transform(frame)
        return frame, 0



import decord
decord.bridge.set_bridge('torch')
import numpy as np
from decord import  AudioReader, cpu
from torch.utils.data.dataset import Dataset

class DecordAudioDataset(Dataset):
    def __init__(self, file_path, transform=None, step_size_s=1, chunk_size_s=2, v=False,  pad=False):
        self.file_path= file_path
        self.ar = AudioReader(self.file_path, ctx=cpu(0), mono=True, #default setting 
            sample_rate=48000) # enforce this sample rate, and convert if nessesary
        #ar.get_info()
        # make sure its mono
        assert np.all(self.ar._array[0]==self.ar._array.mean(axis=0))
        
        self.audio_sr = 48000;
        self.audio_duration_s = self.ar.duration();
        self.n_audio_samples = self.ar._array.shape[1];
        self.chunk_size=chunk_size_s * self.audio_sr;
        if v: print(self.chunk_size)

        # step_size_s should be TR
        n_tr_samples = (self.n_audio_samples/self.audio_sr)/step_size_s
        #print(n_tr_samples, math.ceil(n_tr_samples), int(n_tr_samples), int(round(n_tr_samples)))
        n_tr_samples = int(round(n_tr_samples))

        tr_starts_s = np.arange(n_tr_samples)*step_size_s
        chunk_end_s = tr_starts_s+step_size_s
        chunk_start_s = chunk_end_s-chunk_size_s
        
        self.chunk_start_idx = np.clip(np.round(chunk_start_s*self.audio_sr), 0, self.n_audio_samples-1).astype(int)
        self.chunk_end_idx   = np.clip(np.round(chunk_end_s  *self.audio_sr), 0, self.n_audio_samples-1).astype(int)
        if v: print(np.unique(self.chunk_end_idx-self.chunk_start_idx, return_counts=1))
        
        self.transform = transform;
        self.pad=pad

    def __len__(self): return len(self.chunk_start_idx)
    
    def __getitem__(self, idx):  
        start = self.chunk_start_idx[idx]
        end = self.chunk_end_idx[idx]
        track = self.ar._array[0, start:end]
            
        if self.transform is None and self.pad:
            padding_needed= self.chunk_size - len(track);
            if padding_needed>0:
                track = np.pad(track, (padding_needed, 0), 'constant', constant_values=0)
        
        track = track.reshape(1, -1)
        raw_track_len=track.shape[-1]
        attn_mask = 0;
        if self.transform is not None:
            prep_track = self.transform(track)
            track=prep_track;
            
            if hasattr(prep_track, "input_features"):
                track = prep_track.input_features[0] 
            if  hasattr(prep_track, "input_values"):
                track = prep_track.input_values[0];
            #print(track.shape)
                
            attn_mask = prep_track.attention_mask[0] \
              if hasattr(prep_track, "attention_mask") else raw_track_len;
        
        return track, attn_mask

##############################################################################
# Multiframe Video datasets

import decord
decord.bridge.set_bridge('torch')
import numpy as np
from decord import VideoReader, cpu
decord.bridge.set_bridge('torch')

from torch.utils.data.dataset import Dataset

class TRClipVideoDecordDataset(Dataset):
    def __init__(self, movie_file_path, target_mri_TR, transform=None, num_threads=1):
        self.videoreader = VideoReader(movie_file_path, num_threads=num_threads, ctx=cpu(0))
        self.videoreader.seek(0)
        self.transform = transform
        self.n_frames = len(self.videoreader)
        self.fps = self.videoreader.get_avg_fps();

        self.n_tr_samples = int(round((self.n_frames/self.fps)/target_mri_TR))
        
        tr_start_s = (np.arange(self.n_tr_samples)*target_mri_TR)

        self.start_frame_idxs = np.round(tr_start_s*self.fps).astype(int)
        # make sure it stays in the bounds
        self.start_frame_idxs = np.clip(self.start_frame_idxs, 0, self.n_frames-1)#.astype(int)
        self.frames_per_tr = int(target_mri_TR*self.fps)

        
    def __len__(self): return len(self.start_frame_idxs)
    
    def __getitem__(self, idx):   
        mvidx = self.start_frame_idxs[idx]
        frames = self.videoreader[mvidx:mvidx+self.frames_per_tr]
        if self.transform is not None:
            frames = self.transform(frames)
        return frames, 0
        

##############################################################################
# Pure text datasets

# Custom dataset to hold the sentences
import string, re

def normalize_pauses(text):
    return re.sub(r'\.{3,8}', '\n', re.sub(r'\.{9,}', '\n\n', text))

class SentenceDataset(Dataset):
    def __init__(self, sentences, mode="last_n_trs", last_n_trs=5, n_used_words=510, prep_sentences=None):
        self.sentences = sentences
        self.prep_sentences = prep_sentences
        if self.prep_sentences=="contpretr-friends-v1":
            self.sentences = [s if not(s is np.nan) else "..." for s in self.sentences]

        self.mode=mode
        self.last_n_trs = last_n_trs;
        self.n_used_words = n_used_words;

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = ""
        if self.mode == "last_n_trs":
          text= self.sentences[idx-self.last_n_trs: idx+1]
          text= "".join(text)

        elif self.mode=="n_used_words":
          tr_text = "".join(self.sentences[:idx+1])
          nopunct_text = tr_text#tr_text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
          text= " ".join(nopunct_text.split(" ")[-self.n_used_words:])

        if self.prep_sentences=="contpretr-friends-v1":
            text = normalize_pauses(text)

        if text=="": text= " "
        return text
    


# Use as:
"""

# text_per_tr ~ list of strings: ["text 1", "text 2", ...]
dataset = SentenceDataset(text_per_tr, mode="n_used_words", n_used_words=150)
dataset = SentenceDataset(text_per_tr, mode="last_n_trs", n_used_words=5)

# Custom collate function to handle tokenization
def collate_fn(batch):
    encoding = tokenizer.batch_encode_plus(batch, padding=True, return_tensors="pt", return_attention_mask=True)
    return encoding['input_ids'], encoding['attention_mask']

dataloader = DataLoader(dataset, batch_size=40, shuffle=False, collate_fn=collate_fn)
embd_data = []

  for input_ids, attention_mask in tqdm(dataloader, total=len(dataloader)):
    
""";


    
##############################################################################
# Get the NSD dataset

from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np

brick2pil = lambda img_data : Image.fromarray(np.uint8(img_data)).convert('RGB')

class ImgDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self): return len(self.images)
    
    def __getitem__(self, idx):   
        image = self.images[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, 0
    
class NSD_DataSet(ImgDataset):
    def __init__(self, subset="nsd2536rep3"):
        import h5py
        import pandas as pd
        from tqdm.auto import tqdm
        
        stimuli_file = "/scratch/users/robert.scholz2/nsd/nsd_stimuli.hdf5"
        sf = h5py.File(stimuli_file, 'r')
        sdataset = sf.get('imgBrick')
        
        # subset == "nsd2536rep3"
        fn = f"/scratch/users/robert.scholz2/nsd/subj01/subj01_fsav_fhrfGLMdenoise_betas_20sess_reliable_verts.npy"
        pk = np.load(fn, allow_pickle=True).item()
        img_ids = pk["image_ids"] -1
    
        # if subset == "nsd2000"
        #s1r = pd.read_csv("misc/nsd_subj1resp.tsv", delimiter="\t")
        #img_ids = np.asarray(s1r['73KID']) -1 
        
        if subset == "nsd907":
          nsd_dir ="/scratch/users/robert.scholz2/nsd"
          common_stim = np.load(f"{nsd_dir}/common_images_across_all_8_subjs.npy")
          img_ids = np.asarray(common_stim).astype(int) -1 

        
        subj1_pil_imgs = [brick2pil(sdataset[img_ids[i]]) for i in tqdm(range(len(img_ids)))]
        
        from torchvision.transforms import PILToTensor
        pil2frame= lambda p : PILToTensor()(p).moveaxis(0,-1)
        subj1_imgs = [pil2frame(i) for i in subj1_pil_imgs]
        
        super().__init__(subj1_imgs)

##############################################################################
# Pure Audio datasets

from torch.utils.data.dataset import Dataset
from scipy.io import wavfile
import numpy as np

from scipy.io import wavfile


class WavAudioDataset(Dataset):
    def __init__(self, file_path, transform=None, step_size_s=1, chunk_size_s=2, v=False):

        self.file_path= file_path
        self.audio_sr, self.audio_raw = wavfile.read(file_path)

        if self.audio_sr != 48000:
            print(f"Warning: Audio file {file_path} has an sr of {self.audio_sr}, but most likely an sr of 48000 is required, if not handled otherwise")
        # convert to mono
        if len(self.audio_raw.shape)>1:
          self.audio_raw= np.mean(self.audio_raw, axis=-1)
        
        self.n_audio_samples = self.audio_raw.shape[0]
        self.chunk_size= int(chunk_size_s * self.audio_sr);
        if v: print(self.chunk_size)

        # step_size_s should be TR
        n_tr_samples = (self.n_audio_samples/self.audio_sr)/step_size_s
        n_tr_samples = int(round(n_tr_samples))
        tr_starts_s = np.arange(n_tr_samples)*step_size_s
        chunk_end_s = tr_starts_s+step_size_s
        chunk_start_s = chunk_end_s-chunk_size_s
        
        self.chunk_start_idx = np.clip(np.round(chunk_start_s*self.audio_sr), 0, self.n_audio_samples-1).astype(int)
        self.chunk_end_idx   = np.clip(np.round(chunk_end_s  *self.audio_sr), 0, self.n_audio_samples-1).astype(int)
        if v: print(np.unique(self.chunk_end_idx-self.chunk_start_idx, return_counts=1))
        
        self.transform = transform;
        
    def __len__(self): return len(self.chunk_start_idx)
    
    def __getitem__(self, idx):  
        start = self.chunk_start_idx[idx]
        end = self.chunk_end_idx[idx]
        #track = self.ar._array[0, start:end]
        track = self.audio_raw[start:end]
        
        if self.transform is None:
            padding_needed= self.chunk_size - len(track);
            if padding_needed>0:
                track = np.pad(track, (padding_needed, 0), 'constant', constant_values=0)
        
        rawtrack = track.reshape(1, -1)
        attn_mask = 0;
        if self.transform is not None:
            prep_track = self.transform(rawtrack)
            track=prep_track;
            
            if hasattr(prep_track, "input_features"):
                track = prep_track.input_features[0] 
            if  hasattr(prep_track, "input_values"):
                track = prep_track.input_values[0];
            #print(track.shape)
                
            attn_mask = prep_track.attention_mask[0] if hasattr(prep_track, "attention_mask") else rawtrack.shape[-1];
        
        return track, attn_mask





from torch.utils.data.dataset import Dataset
from scipy.io import wavfile
import numpy as np

class MultiFilesWavAudioDataset(Dataset):
    def __init__(self, stimuli_paths, transform=None,\
                 num_threads=1, require_sr=44100, use_tqdm=True):

        raw = []
        
        loop_over = stimuli_paths;
        if use_tqdm:
            from tqdm.auto import tqdm;
            loop_over = tqdm(stimuli_paths)
    
        for path in loop_over:
            #print(path)
            sr, audio_raw = wavfile.read(path)
            assert sr == require_sr;
            raw.append(audio_raw.astype(np.float32))
        
        self.audio_data = raw
        self.audio_sr = require_sr;
        self.transform = transform;
        
    def __len__(self): return len(self.audio_data)
    
    def __getitem__(self, idx):   
        track = self.audio_data[idx]
        track = track.reshape(1, -1)
        if self.transform is not None:
            track = self.transform(track)
        return track, 0
    


