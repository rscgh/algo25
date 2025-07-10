import torch
from videomae_friends_dataset import FriendsDatasetTest
import videomae
import numpy as np
from sklearn.utils import Bunch
import os
from brainannlib.algonauts_funcs import compute_encoding_accuracy
from tqdm.auto import tqdm

# opts
#shared/barbano/videomae_mae_downsampled__nframes16_sub1235_idTrue_w5_hrf2_fmriW1_adamw_lr0.0001_decaystep_wd1e-05_bsz64_ts1_epochs100_s0.pth
#nframes16_
# idTrue_w5_hrf2_fmriW1_adamw_lr0.0001_decaystep_wd1e-05_bsz64_ts1_epochs100_s0.pth

opts=Bunch(**{})
opts.stimulus_window = 5
opts.hrf_delay = 2
opts.n_frames=16
opts.fmri_window=1
opts.model = "videomae_mae"
opts.device="cuda"
opts.encode_subject_id=True
opts.subjects=[1,2,3,5]
opts.method="mae"
opts.downsampled=True
opts.data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
opts.timesample=1
opts.test_seasons=["figures"]
opts.batch_size=64
opts.amp=True

chkpt_file = "/scratch-scc/users/robert.scholz2/cneuromod/videomae_cb.pth"

def load_model(opts):
    #if opts.model == "videomae":
    model = videomae.VideoMAERegression(criterion=opts.method, use_subject_id=opts.encode_subject_id).to(opts.device)
    return model.image_processor(), model

preprocess, model = load_model(opts)
model.eval()

test_dataset = FriendsDatasetTest(root=opts.data_dir, timesample=opts.timesample, image_transform=preprocess,
                                  downsampled=opts.downsampled, stimulus_window=opts.stimulus_window,
                                  hrf_delay=opts.hrf_delay, subjects=opts.subjects, seasons=opts.test_seasons,
                                  target_video_len=opts.n_frames)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False,
                                                  num_workers=4, pin_memory=True,
                                                  prefetch_factor=2)


checkpoint = torch.load(chkpt_file, map_location=opts.device, weights_only=False)
model.load_state_dict(checkpoint['model'])
start_epoch = checkpoint['epoch'] + 1
print(f"Restored model from epoch {start_epoch}")

#r, test_loss, _, _ = test(model, test_dataloader, opts, epoch, writer, scaler)
all_outputs=[]
all_subjects=[]
all_mri=[]
for idx, (video, fmri, subjects) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        video = video.to(opts.device)
        #fmri = fmri.to(opts.device)
        subjects = subjects.to(opts.device)

        video['pixel_values'] = video['pixel_values'].squeeze(1)
        bsz = video['pixel_values'].shape[0]

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=opts.amp):
                #_, outputs = model(video, fmri, subjects.half())
                video_features = model.encode_video(video)
                outputs = model.predict_activation(video_features, subjects.half())

        all_outputs.append(outputs.detach().cpu().numpy())
        #all_labels.append(fmri)
        all_subjects.append(subjects.int())
        all_mri.append(fmri.detach().cpu().numpy())

all_outputs=np.concatenate(all_outputs,axis=0)
all_subjects=torch.concatenate(all_subjects,axis=0).cpu().detach().numpy()
all_mri=np.concatenate(all_mri,axis=0)


for subject in np.unique(all_subjects):
    y_test = all_mri[all_subjects==subject]
    y_pred = all_outputs[all_subjects==subject]
    encoding_accuracy, mean_acc = compute_encoding_accuracy(y_test, y_pred, subject, "all")
    print(f"Subject {subject} n={(all_subjects==subject).sum()} corr=", mean_acc.round(4));