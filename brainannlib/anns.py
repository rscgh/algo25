######################################################################################
## This file provides functions for the loading of the different ANNs that 
## we want want to test as models for the brain, along with the nessesary 
## model-specific preprocessing transformations that have to be applied
## to the stimuli prior to passing them through the ANNs
##
## Annotation needs to be improved
######################################################################################

import sys, os
import numpy as np

##############################################################################
# to make models run fast on CPUs

from torch import set_num_threads
# from brainannlib.stats_and_metrics import get_num_assigned_cpus
# set_num_threads(get_num_assigned_cpus())


##############################################################################
# Model loading

def load_model(name, pretrained=True):
    # returns model, transforms, forward;
    
    if name=="AudioClip_audio":
        return load_audio_clip(pretrained = pretrained);
    
    if name=="AST":
        return load_ast(pretrained = pretrained);
    
    if name=="wave2vec":
        return load_wave2vec(pretrained = pretrained);
    
    if name=="clipvit-base-patch32":
        return load_vis_clip(pretrained=pretrained)
        
    if name=="alexnet":
        return load_alexnet(pretrained = pretrained);

    if name=="internViT":
        return load_InternViT_v2(pretrained = pretrained);
    
    if name.startswith("yolo"):
        return load_yolo(pretrained = pretrained, version = name+".pt")

    if name=="whisper-large-v3":
        #return load_whisperv3(pretrained = pretrained)
        return load_whisperv3_vb(pretrained = pretrained)

    if name=="whisper-small":
        return load_whisper_small(pretrained = pretrained)

    return None,None,None;



""";
# Dataset setup for testing of the visual models

import torch
stim_path = '/scratch/users/robert.scholz2/hcp_movies/Post_20140821_version/7T_MOVIE2_HO1_v2.mp4'

import sys # for using decord
sys.path.append("/home/mpg02/MLSC/robert.scholz2/.local/lib/python3.7/site-packages")
from lib.data_loading import TRSamplingDecordVDataset
ds2 = TRSamplingDecordVDataset(stim_path, 1.49, transform=None, num_threads=0)
"""


def load_vis_clip(pretrained=True):
    from transformers import CLIPProcessor, CLIPModel, CLIPConfig
    
    if pretrained:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    else:
        cfg = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel(cfg)
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    #preprocess = lambda frame: processor(images=frame, return_tensors="pt", padding=True)["pixel_values"]
    preprocess = lambda frame: processor(images=frame.moveaxis(-1,0), return_tensors="pt", padding=True)["pixel_values"]
    #include_layers_containing = ["vision_model.*[0-9]+.mlp.fc2"] #["fc2"] # "vision_model.*(0|4|8|11).mlp.fc2"

    #model_desc = "CLIPb32"
    model_forward = lambda batch : \
       model.get_image_features(pixel_values = batch.squeeze((1)))
    
    """
    from transformers.image_utils import ChannelDimension
    a = cprocessor(images=frame, return_tensors="pt", padding=True)['pixel_values']
    b = cprocessor(images=frame, return_tensors="pt", padding=True, data_format=ChannelDimension.LAST)['pixel_values']
    print(frame.shape)
    print(a.shape, b.shape)""";
    return model, preprocess, model_forward


"""
# Verify clip input:

model, transforms, forward = load_vis_clip(pretrained=True)

frame = ds2.videoreader[ds2.frame_idxs[410]]

# to show the image of geoge clooney + screen:
to_pil(frame.moveaxis(-1,0))

inp = transforms(frame)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
inputs = processor(text=["a photo of a man", "a photo of a woman"], images=to_pil(frame.moveaxis(-1,0)), return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs) # tensor([[0.9946, 0.0052]], device='cuda:0', dtype=torch.float16) # -> correctly identifies the man

# now we just have to see if
# model(**inputs) 
# delivers the same result as our forward of only the image data
# model.get_image_features(pixel_values = batch)

outputs = model(**inputs)
#a = outputs.vision_model_output.last_hidden_state
#print(a.shape, a.flatten()[:4])
a = outputs.image_embeds
print("#", a.shape, a.flatten()[:4])

outputs = forward(inp)
print(outputs.shape, outputs.flatten()[:4])
outputs = forward(inputs["pixel_values"])
print(outputs.shape, outputs.flatten()[:4])
forward = lambda batch : model.get_image_features(pixel_values = batch)
outputs = forward(inp)
print(outputs.shape, outputs.flatten()[:4])
a = outputs / _get_vector_norm(outputs)
print("#", a.shape, a.flatten()[:4])

"""



def load_alexnet(pretrained=True):
    from PIL import Image
    from torchvision import transforms
    
    import torch
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=pretrained)
    model.eval()

    processor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    #input_tensor = preprocess(input_image)
    #input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    import torchvision.transforms as T
    to_pil = T.ToPILImage()
    preprocess = lambda frame : processor(to_pil(frame.moveaxis(-1,0)))#.unsqueeze(0)
    #preprocess = lambda frame_tensor : processor(to_pil(frame_tensor.moveaxis(-1,0))).unsqueeze(0)
    
    model_forward = lambda batch : model(batch)
    
    return model, preprocess, model_forward;
    
"""
Testing of alexnet:


from lib.anns import load_alexnet
model, transforms, forward = load_alexnet(pretrained=True)

with open('imagenet_classes.txt') as f:
  labels = [line.strip() for line in f.readlines()]

import torchvision.transforms as T
to_pil = T.ToPILImage()

frame = ds2.videoreader[ds2.frame_idxs[410]]
print(frame.shape)
print(ds2.transform(frame).shape)
out = forward(transforms(frame).unsqueeze(0))
print(out.shape)
_, index = torch.max(out, 1)
print(index) # should be tensor([598])
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item())
# should be: home theater, home theatre 26.5106143951416
to_pil(frame.moveaxis(-1,0))
"""
    
    
"""
import sys
sys.path.append("/home/mpg02/MLSC/robert.scholz2/.local/lib/python3.7/site-packages")

ds = DecordVDataset(movie_file_path, preprocess)
stimulus_loader = DataLoader(ds, batch_size = int(round(len(ds)/(desired_n_chunks))), num_workers=0)
for inp_batch, labels in t:
    output = model_forward(inp_batch)

ds = TRSamplingDecordVDataset(movie_file_path, target_mri_TR, transform=preprocess, num_threads=0)
stimulus_loader = DataLoader(ds, batch_size = 5)
for inp_batch, labels in t:
    output = model_forward(inp_batch)
    
# try
model.apply(model._init_weights);
#model_desc = model_desc + "_Untrained"
print(model_desc)

"""

def load_resnet50(pretrained=True):
    from torchvision.models import resnet50, ResNet50_Weights
    
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None;
    model = resnet50(weights=weights);
    transforms = ResNet50_Weights.IMAGENET1K_V2.transforms()
    
    import torchvision.transforms as T
    to_pil = T.ToPILImage()
    preprocess = lambda frame : transforms(to_pil(frame.moveaxis(-1,0)))#.unsqueeze(0)
    
    model_forward = lambda batch : model(batch)
    return model, preprocess, model_forward;

"""

model, preprocess, forward = load_resnet50();

inp = preprocess(frame).unsqueeze(0);
out = forward(inp)
print(out.shape)
prediction = out.squeeze(0).softmax(0)
class_id = prediction.argmax().item()
print(class_id)
score = prediction[class_id].item()

from torchvision.models import ResNet50_Weights
weights = ResNet50_Weights.IMAGENET1K_V2
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

"""





def load_yolo(pretrained=True, version = "yolov8n.pt"):
    from ultralytics import YOLO, settings
    #print(settings)
    
    
    path = f"/usr/users/robert.scholz2/.cache/{version}"
    if not(os.path.exists(path)):
        print(f"please download model version {version} to \n ... {path}")
        print(f"e.g. via: !wget -O /usr/users/robert.scholz2/.cache/{version} " + \
              f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{version}")
        return None, None, None;

    model = YOLO(path)
    #model.eval()
    import torchvision.transforms as T
    to_pil = T.ToPILImage()
    preprocess = lambda frame : frame #to_pil(frame.moveaxis(-1,0))    
    #model_forward = lambda batch:  model([to_pil(x.moveaxis(-1,0)) for x in batch], verbose=False)

    
    if not pretrained:
        model.reset_weights();
        model.model.eval();
        
    model_forward = lambda batch: model([to_pil(x.moveaxis(-1,0)) for x in batch], verbose=False)

    
    return model, preprocess, model_forward;



"""from huggingface_hub import hf_hub_download
cfgfile = hf_hub_download(repo_id="OpenGVLab/InternViT-6B-224px", filename="preprocessor_config.json")
from transformers import CLIPFeatureExtractor
fe = CLIPFeatureExtractor.from_json_file(cfgfile)""";

from transformers.dynamic_module_utils import get_imports
def fixed_get_imports(filename):
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
        print("ran-fix:", imports)
    return imports

def load_InternViT_v2(pretrained=True):
    from transformers import AutoModel, CLIPImageProcessor
    from unittest.mock import patch
    import torch
    
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModel.from_pretrained('OpenGVLab/InternViT-6B-448px-V1-2', torch_dtype=torch.bfloat16, \
                                    trust_remote_code=True, use_flash_attn =False).eval()
    if not pretrained:
        model = AutoModel.from_config(model.config).eval();
    
    image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-6B-448px-V1-2')
    preprocess = lambda frame: image_processor(images=frame.moveaxis(-1,0), return_tensors="pt")["pixel_values"]
    model_forward = lambda batch : model(batch.squeeze((1)).to(model.dtype))

    return model, preprocess, model_forward;






"""
General test config

from lib.data_loading import DecordAudioDataset
file_path = "/scratch/users/robert.scholz2/hcp_movies/Post_20140821_version/7T_MOVIE2_HO1_v2.mp4"
ds = DecordAudioDataset(file_path, transform=None, num_threads=0, step_size_s=1, chunk_size_s=5, use_tqdm=1)


from torch.utils.data.dataloader import DataLoader
stimulus_loader = DataLoader(ds, batch_size = 30, num_workers=0)
"""

"""
first in your current directory (or any): 
mkdir utils
cd ./utils
git clone https://github.com/AndreyGuzhov/AudioCLIP.git
wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt -O data/weights/AudioCLIP-Full-Training.p

"""

def load_audio_clip(pretrained=True):
    # needed for torch vision
    #tv_path = "/home/mpg02/MLSC/robert.scholz2/.local/lib/python3.7/site-packages"
    #if not(tv_path in sys.path): sys.path.append(tv_path)
    # needed for AudioCLIP
    ac_root= os.path.abspath(f'{os.getcwd()}/external/utils/AudioCLIP')
    print(ac_root)
    if not(ac_root == sys.path[-1]): sys.path.append(ac_root)
        
    from model import AudioCLIP
    
    import torchvision as tv
    class ToTensor1D(tv.transforms.ToTensor):
      def __call__(self, tensor: np.ndarray):
        tensor_2d = super(ToTensor1D, self).__call__(tensor[..., np.newaxis])
        return tensor_2d.squeeze_(0)
    
    #from utils.transforms import ToTensor1D
    # from lib.data_loading

    # uses 44k as sample rate apparently, so no need for much transforms
    #https://github.com/AndreyGuzhov/AudioCLIP/blob/master/demo/AudioCLIP.ipynb
    
    pt = f'data/weights/AudioCLIP-Full-Training.pt' if pretrained else False;
    model = AudioCLIP(pretrained=pt)
    model.eval()
    audio_transforms = ToTensor1D()
    model_forward = lambda audio_inp, sr=None : model(audio=audio_inp)
    
    return model, audio_transforms, model_forward;




def load_wave2vec(pretrained=True):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    
    # large version: "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    # small version: "facebook/wav2vec2-base-960h" 
    MODEL_ID = "patrickvonplaten/wav2vec2-base-100h-2nd-try" 
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    #model_input_sr = 16000

    import torchaudio.transforms as T
    from torch import tensor
    resample = T.Resample(44100, 16000)
    
    preprocess = lambda audio: processor(resample(tensor(audio[0]).float()), sampling_rate=16000, return_tensors="pt").input_values.squeeze(axis=0)
    #preprocess = lambda audio_chunk: processor(audio_chunk, sampling_rate=16000, return_tensors="pt").input_values.squeeze(axis=0)
    
    model_fwd_kwargs = dict()
    model_forward = lambda audio_inp, sr=None: model(audio_inp)
    
    return model, preprocess, model_forward



def load_ast(pretrained=True):
    
    from transformers import AutoProcessor, ASTForAudioClassification
    model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").eval()
    
    if not pretrained:
        model = ASTForAudioClassification(model.config).eval()
    
    processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    #resample = T.Resample(stimulus_loader.dataset.audio_sr, 16000)
    import torchaudio.transforms as T
    from torch import tensor
    resample = T.Resample(44100, 16000)
    transform = lambda audio : processor(resample(tensor(audio[0]).float()), sampling_rate=16000).input_values[0]
    model_forward = lambda audio_inp : model(input_values=audio_inp)
    
    return model, transform, model_forward; 



"""
from lib.anns import load_ast
model, transform, model_forward = load_ast(True)

stimulus_loader.dataset.transform = transform
batch, _ = next(iter(stimulus_loader))
print(batch.shape)
o=model(input_values=batch)
o.keys(), o.last_hidden_state.shape, o.pooler_output.shape

from transformers import AutoProcessor, ASTModel
import torch

model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

from datasets import load_dataset
dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)


"""




def load_whisperv3_vb(pretrained=True):

    # model specific imports
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import torchaudio.transforms as T
    from torch import tensor
    
    # load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").eval()
    
    if not pretrained:
        model = WhisperForConditionalGeneration(model.config).eval()

    # twice sends a stronger singal to prepreae starting a new sentence
    # and could give more information in the end
    #decoder_input_ids = tensor([[1, 1]]) * model.config.decoder_start_token_id
    transform = lambda x : processor(T.Resample(44100, 16000)(tensor(x[0]).float()), sampling_rate=16000).input_features[0]
    #model_forward = lambda audio_inp : model(audio_inp, decoder_input_ids=decoder_input_ids)
    model_forward = lambda audio_inp : model.generate(audio_inp, max_new_tokens=1)
    return model, transform, model_forward



def load_whisper_small(pretrained=True):

    # model specific imports
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import torchaudio.transforms as T
    from torch import tensor
    
    # load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").eval()
    
    if not pretrained:
        model = WhisperForConditionalGeneration(model.config).eval()

    # twice sends a stronger singal to prepreae starting a new sentence
    # and could give more information in the end
    #decoder_input_ids = tensor([[1, 1]]) * model.config.decoder_start_token_id
    transform = lambda x : processor(T.Resample(44100, 16000)(tensor(x[0]).float()), sampling_rate=16000).input_features[0]
    #model_forward = lambda audio_inp : model(audio_inp, decoder_input_ids=decoder_input_ids)
    model_forward = lambda audio_inp : model.generate(audio_inp, max_new_tokens=1)
    return model, transform, model_forward
    



