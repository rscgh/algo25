from nilearn import datasets, plotting
from matplotlib import pyplot as plt;
import numpy as np;
import io, PIL, os

def fig2pil(fig, close_fig = True):
  buf = io.BytesIO()
  fig.savefig(buf, bbox_inches="tight", pad_inches=0.02);
  buf.seek(0)
  img = PIL.Image.open(buf)
  if close_fig: plt.close(fig) 
  return img

#####################################################################################################
# Stats diagram plotting

def barplot_model_scores_hor(datadicts = None, n_comps = 20, baseline=None, vlim=[0, 0.34], ax=None, by_layer=True, \
      figsize=(5,4), norm_by_bl=False, clabels=None, remove_legend=True, colors=None, hatch='////', estimator="mean", 
                             bl_prep=lambda x: np.mean(x, 0), print_scores=False, pnames=None, **kwargs):
    
    import seaborn as sns
    import pandas as pd
    #'#ffdddd', '#ffdddd'
    if colors is None:
        colors = ["#00BBEB", "#37899E", "#DDD"] # trained, untrained, baseline
        colors = ["#59a89c", "#f0c571", "#DDD"]
    
        
    norm = bl_prep(baseline["scores"][:,:n_comps]) if norm_by_bl else 1    
    vn = "adj_r" if norm_by_bl else 'r';
    
    if ax is None: ax = plt.figure(figsize=figsize).gca();
        
    if not(baseline is None):
        x = bl_prep(baseline["scores"][:,:n_comps]) / norm
        dfbl = pd.DataFrame(x)
        dfbl = dfbl.T.melt(var_name='comp', value_name=vn)
        sns.barplot(x=vn, y='comp', data=dfbl, color="#DDD", ax=ax, orient="h" , estimator="mean")#,  hatch='////', edgecolor="#999", linewidth=0)
        
    first_df=None
    for i, datad in enumerate(datadicts): 
        #print(i)
        #x = datad["scores"][:,:n_comps].mean(0) / norm
        x = datad["scores"][:,:n_comps] / norm
        df = pd.DataFrame(x)
        df['layer'] = range(len(df))
        df_melted = df.melt(id_vars='layer', var_name='comp', value_name=vn)
        xargs = dict(hatch=hatch, edgecolor=colors[i-1], linestyle="none") if i==1 else {}; #edgecolor=(0,0,0,0)
        xargs.update(kwargs)
        sns.barplot(x=vn, y='comp', data=df_melted, color=colors[i], ax=ax, orient="h", estimator=estimator, **xargs)
        #if i==len(datadicts)-1: first_df = df_melted.copy();
        if i==0: first_df = df_melted.copy();

        if print_scores:
            from IPython.display import display, HTML
            print("Dataframe: " + ("" if pnames is None else pnames[i]))
            mean = df.mean(axis=0).to_list()[:-1] + ["mean"]
            mmax = df.max(axis=0).to_list()[:-1] + ["max"]
            mmin = df.min(axis=0).to_list()[:-1] + ["min"]
            df.loc[len(df)] = mean #list(randint(10, size=2))
            df.loc[len(df)] = mmax #list(randint(10, size=2))
            df.loc[len(df)] = mmin #list(randint(10, size=2))

            display(HTML(df.round(2).to_html()))
            #display()
                
    if by_layer:
        
        sns.stripplot(x=vn, y='comp', data=first_df, hue='layer', palette='inferno', jitter=True, ax=ax, orient="h",legend = 'full')
        
        if remove_legend:
            ax.get_legend().remove()
            
        legend= ax.get_legend()
        handles = legend.legend_handles
        labels = [int(x._text)+1 for x in legend.texts]
        # Create a list of indices to keep
        indices = np.linspace(0, len(handles) - 1, 5, dtype=int)
        # Keep only the selected handles and labels
        handles = [handles[i] for i in indices]
        labels = [labels[i] for i in indices]
        # Create the new legend
        ax.legend(handles, labels, loc='upper right', title="Layer")
        
    ax.set_xlim(*vlim)
    ax.set_ylabel("")
    
    #if :    
    ax.set_yticks(np.arange(n_comps))
    ax.set_yticklabels(clabels[:n_comps] if not(clabels is None) else np.arange(n_comps)+1)
    
    #plt.tight_layout()



#####################################################################################################
# Flatmap plotting

import os
root_data_dir = os.environ["ALGONAUTS_ROOT_DIR"]
import cortex
cortex.database.default_filestore = f"{root_data_dir}/pycortex_filestore"
cortex.db.filestore = f"{root_data_dir}/pycortex_filestore"
cortex.db = cortex.database.Database(f"{root_data_dir}/pycortex_filestore")
#print(cortex.database.default_filestore)

def plot_flatmap(data, mask=None, mtype = "fsaverage", height=250, data_style=None, overlay_file="", roi_list=[], qkw={}, roikw={}, **kwargs):
  vdata = data
  if not(mask is None):
    vdata = np.zeros_like(mask)
    vdata[mask] = data
  
  if data_style=="fullHCP":
    import hcp_utils as hcp
    vdata = np.concatenate([hcp.left_cortex_data(vdata), hcp.right_cortex_data(vdata)]);
    #fig = cortex.quickshow(vertex_data, with_rois=False, height =1300, shadow=10,  )
  
  vertex_data = cortex.Vertex(vdata, mtype,**kwargs)
  qsparams=dict( with_rois=0, height =height, colorbar_location=(0.4,-0.08,0.2,0.05));
  qsparams.update(qkw)
  fig = cortex.quickshow(vertex_data, **qsparams);
  if roi_list !=[]: 
      roi_def_kw={"stroke":"#999", "label-font-size":"16pt"}
      roi_def_kw.update(roikw)
      img, im, selfobj = add_rois(fig, vertex_data, overlay_file=overlay_file, roi_list=roi_list, **roi_def_kw)
  return fig;


def plot_layer_predictions_flat(scores, mask = None, mtype="fsaverage", auto_max=None):
   n=len(scores)
   fig, axs = plt.subplots(1,n, figsize=(5*n,3))
   vmin, vmax = (None, None) if (auto_max is None) else (0, auto_max * np.array([s for k, s in scores.items()]).max())
   i=0
   for layer, score_data in scores.items():
       fig2 = plot_flatmap(score_data, mask, mtype=mtype, vmin = vmin, vmax=vmax)
       axs[i].imshow(fig2pil(fig2)); axs[i].set_title(layer); axs[i].axis("off");
       i+=1;
   
   plt.subplots_adjust(wspace=0, hspace=0);
   plt.tight_layout();
   return fig;


## -----------------------------------------------------------
## Own extension of flatmap plotting with ROIs

from lxml import etree
import cairosvg
from lxml.builder import E
from cortex.svgoverlay import Overlay
import tempfile
import copy

def own_get_texture(selfobj, layer_name, height, name=None, background=None, labels=True,
        shape_list=None, **kwargs):

        import matplotlib.pyplot as plt
        # Set the size of the texture
        if background is not None:
            img = E.image(
                {"{http://www.w3.org/1999/xlink}href":"data:image/png;base64,%s"%background},
                id="image_%s"%name, x="0", y="0",
                width=str(selfobj.svgshape[0]),
                height=str(selfobj.svgshape[1]),
            )
            selfobj.svg.getroot().insert(0, img)
        if height is None: height = selfobj.svgshape[1]
        height = int(height)
        
        # separate kwargs starting with "label-"
        label_kwargs = {k[6:]:v for k, v in kwargs.items() if k[:6] == "label-"}
        kwargs = {k:v for k, v in kwargs.items() if k[:6] != "label-"}

        for layer in selfobj:#svg.layers.values():
            #print(layer)
            if layer.name == layer_name:
                #print("Found", layer_name)
                layer.visible = True
                layer.labels.visible = labels
                for name_, shape_ in layer.shapes.items():
                    # honor visibility set in the svg
                    if shape_list is not None:
                        shape_.visible = name_ in shape_list
                    #print(name_, shape_.visible)
                    # Set visibility of labels (by setting text alpha to 0)
                    # This could be less baroque, but text elements currently
                    # do not have individually settable visibility / style params
                    tmp_style = copy.deepcopy(layer.labels.text_style)
                    tmp_style['fill-opacity'] = '1' if shape_.visible else '0'
                    # {'font-family': 'Helvetica, sans-serif', 'font-size': '14pt', 'font-weight': 'bold', 'font-style': 'italic', 'fill': 'white', 'fill-opacity': '1',
                    # 'text-anchor': 'middle', 'filter': 'url(#dropshadow)', 'display': 'inline'}
                    # {'font-size': '20pt'}
                    tmp_style.update(label_kwargs)
                    tmp_style_str = ';'.join(['%s:%s'%(k,v) for k, v in tmp_style.items() if v != 'None'])
                    for i in range(len(layer.labels.elements[name_])):
                        layer.labels.elements[name_][i].set('style', tmp_style_str)
                layer.set(**kwargs)
            else:
                layer.visible = False
                layer.labels.visible = False
        
        
        svg_data = etree.tostring(selfobj.svg)
        png_data = cairosvg.svg2png(bytestring=svg_data)
        png_io = io.BytesIO(png_data)
        im = plt.imread(png_io)
        return im, selfobj

#from cortex.quickflat import _get_extents
from cortex.quickflat.utils import _get_height, _get_extents, _convert_svg_kwargs, _get_fig_and_ax, _parse_defaults
from cortex.database import db
def add_rois(fig, dataview, extents=None, height=None, with_labels=True, roi_list=None, overlay_file=None, **kwargs):
    if extents is None:
        extents = _get_extents(fig)
    if height is None:
        height = _get_height(fig)        
    svgobject = db.get_overlay(dataview.subject, overlay_file=overlay_file)
    im, selfobj = own_get_texture(svgobject, 'rois', height, labels=with_labels, shape_list=roi_list, **kwargs)
    
    _, ax = _get_fig_and_ax(fig)
    img = ax.imshow(im, aspect='equal', interpolation='bicubic',  extent=extents, label='rois', zorder=1000)
    return img, im, selfobj





#####################################################################################################
# 3D Surface plotting



def plot_fsav(vdata, resolution="fsaverage7", **kwargs):
		fsaverage = datasets.fetch_surf_fsaverage(mesh =resolution)
		# vdata should be of shape vdata[:163842]
		pkwargs = dict(roi_map=vdata, hemi='left', cmap="coolwarm", colorbar=True, \
    view='lateral',bg_map=fsaverage['sulc_left'], bg_on_data=True, darkness=.25)
		pkwargs.update(kwargs);
		im = plotting.plot_surf_roi(fsaverage['infl_left'], **pkwargs); # option: pial_left
		#plt.tight_layout();
		return im


# FSLR32k

import os
import nilearn, brainspace
from nilearn import plotting
import hcp_utils as hcp
from matplotlib import pyplot as plt


# new even more general version
stub = os.path.dirname(brainspace.__file__) + "/datasets/surfaces/"

def plot_surface(data, surfs = None, transform=None, title="", \
      plot_conf = ["img1_0_180_6.4","img1_0_0_6.4", "img2_0_0_6.4", "img2_0_180_6.4"], \
      figsize=(19,6), tsize="xx-large", tweight="bold", threshold=1e-14, cmap= "viridis", \
      show_fig=True, tight=True, **kwargs):
    
    if not isinstance(data, dict):
        data = {f"img{i+1}": d for i, d in enumerate(data)} if isinstance(data, list) else {"img1": data}
    
    # data transformations
    if transform in ["29to32k"]:
      transform = {"img1": hcp.left_cortex_data, "img2": hcp.right_cortex_data}
    if transform in ["54to32k"]:
      transform = {"img1" : lambda img : hcp.left_cortex_data(img[hcp.struct.cortex_left]), \
                   "img2" : lambda img : hcp.right_cortex_data(img[hcp.struct.cortex_right])}
          
    # surfaces
    default_conte = {'img1': stub+"conte69_32k_lh.gii" , 'img2' : stub+"conte69_32k_rh.gii" }
    surfs = default_conte if surfs is None else surfs;
    if isinstance(surfs, list):
      surfs = {"img"+str(i+1): surfs[i] for i in range(len(surfs))}
    elif not isinstance(surfs, dict):
      surfs = {"img1" : surfs};
    
    # plotting
    n_plots = len(plot_conf);
    fig, ax = plt.subplots(1,n_plots, figsize=figsize, subplot_kw={'projection': '3d'})
    
    for  n, pcf in enumerate(plot_conf):
      imgid, elev, azim, dist = pcf.split("_");
      elev = int(elev); azim = int(azim); dist=float(dist);
      
      can_transf = not(transform is None)
      t_img_data = transform[imgid](data[imgid]) if can_transf else data[imgid];
     
      plotting.plot_surf_roi(surfs[imgid], roi_map=t_img_data, cmap=cmap, hemi="left", \
            view="medial",darkness=.5, axes=ax[n], threshold=threshold, **kwargs);
      ax[n].view_init(elev, azim)
      ax[n].dist = dist

    y_title_pos = ax[0].get_position().get_points()[1][1]+(1/1)*0.05
    fig.suptitle(title, y=y_title_pos, size=tsize, weight=tweight)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    if tight: fig.tight_layout();
    if show_fig: fig.show()
    return fig


def centered_minmax(data, p = 1):
  vmax = np.max((np.absolute(data.min()),np.absolute(data.max())))
  return -vmax*p, vmax*p;      

# this is still in mm
def plot_sep_colorbar(vmin=0, vmax=1, cmap = "viridis", orientation = "horizontal", path = None, caspect=20, cshrink=1, cfraction=0.15):
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
  img = plt.imshow(np.array([[0,1]]), aspect=0.000000000001)
  plt.gca().axis('off');
  plt.colorbar(sm, orientation=orientation, aspect=caspect, fraction=cfraction, shrink=cshrink);
  if not(path is None):
     plt.savefig(path)

def plot_29k(data, plot_cbar = True, cbar_kwargs = {}, **kwargs):
  vmin, vmax=centered_minmax(data)
  params = dict(vmin=vmin, vmax=vmax, tsize="large", cmap="coolwarm", threshold=None,  transform="29to32k", \
    plot_conf = ["img1_0_180_0","img1_0_0_0"], colorbar=False, figsize=(5,3)); 
  params.update(kwargs)
  fig=plot_surface([data], **params);
  
  if plot_cbar: 
    cbargs = dict(cshrink=0.4, cfraction=0.03, cmap=params["cmap"], caspect=20)
    cbargs.update(cbar_kwargs);
    plot_sep_colorbar(params["vmin"],params["vmax"], **cbargs); 
    plt.tight_layout(); plt.show()
    
  return fig;


from matplotlib import colors

def shifted_cmap(cmap, vmin=-0.2, vmax=0.4, data=None, midpoint=0, greymidpoint=True):
    if not(data is None):
        vmin=data.min(); vmax=data.max();

    new_indices = np.linspace(0, 1, 256)
    zero_point = (midpoint - vmin) / (vmax - vmin)
    shifted_indices = np.interp(new_indices,  [0, zero_point, 1], [0, 0.5, 1])
    colorlist=cmap(shifted_indices)
    if greymidpoint:
        zero_index=np.argmin(np.absolute(shifted_indices-0.5))#+1
        colorlist[zero_index-1:zero_index+1]=(0.8,0.8,0.8, 1)

    new_cmap = colors.ListedColormap(colorlist)
    return new_cmap, vmin, vmax




#####################################################################################################
# 2D color plotting helper
from PIL import Image
from brainannlib.visualization import cortex
import matplotlib

def get_2d_color_from_img(x,y, bounds = [(0,1),(0,1)], cmap_img = None):
  w,h = cmap_img.width, cmap_img.height; 
  rx = ((x-bounds[0][0])/(bounds[0][1]-bounds[0][0]))
  ry = ((y-bounds[1][0])/(bounds[1][1]-bounds[1][0]))
  #print(x,y,rx,ry, round(rx*w), round(ry*h))
  r, g, b, a = cmap_img.getpixel((max(0,min(round(rx*w),w-1)), max(0,min(round(ry*h),h-1))))
  #hex = matplotlib.colors.rgb2hex((r/255,g/255,b/255))
  return (r/255, g/255, b/255, a/255)

# other cmap options: cmapxxx, plasma_alpha
def prepare_2d_colorplot(map1, map2, bounds = None, cmap_img="cmapbyr", v=False):

  if isinstance(cmap_img, str):
    cmapdir=cortex.options.config.get("webgl", "colormaps")
    im = Image.open(f"{cmapdir}/{cmap_img}.png")
    cmap_img = im.convert('RGBA').rotate(270)

  assert len(map1)==len(map2)
  if bounds is None:
    max_value=max(map1.max(), map2.max())
    min_value=max(0,min(map1.min(), map2.min()))
    bounds = [(min_value, max_value), (min_value, max_value)]
    if v: print("Bounds:", min_value, max_value)

  data = np.arange(len(map1))+1;
  c = [get_2d_color_from_img(b,a,cmap_img = cmap_img, bounds=bounds) for a,b in zip(map1,map2) ]
  cmap = matplotlib.colors.ListedColormap([(0.0,0.0,0.0)]+c)
  return data, cmap; 



#####################################################################################################
def cont_any(item, listb):
    for e in listb: 
        if e in item: 
            return True

    return False;
    
def list_existing_files(datasets = None, modes = None, excl_modes=[], show_untr=True):

    regr = {}
    from termcolor import colored
    from lib.config import model_names,dataset_subjects
    from lib.config import datasets_paths, activations_path, scores_path
    from glob import glob
    
    if datasets is None:
        from lib.config import datasets as all_ds
        datasets = all_ds
    
    for dataset in datasets:
        #print(colored(dataset,attrs=["bold"]), "...")
        regr[dataset]= []
        for modeln in model_names: 
            #print(modeln)
            
            df=[]
            targmodes = modes;
            if modes is None:
                fxs = [".".join(f.split("-")[-1].split(".")[:-1]) for \
                       f in glob(f"{scores_path}/{modeln}-*-{dataset}-*")]
                targmodes = np.unique(fxs);
                
            
            #print(targmodes)
            targmodes = [m for m in targmodes if not(cont_any(m, excl_modes))]
            #print(targmodes)

            for mode in targmodes: 

                trainstats = ["pretr", "untr"] if show_untr else ["pretr"];
                for train in trainstats:
                    prefix=f"{modeln}-{train}-{dataset}";
                    #print("-", prefix)
                    files = glob(f"{scores_path}/{prefix}-*{mode}.npy")

                    ds_subjs = dataset_subjects[dataset]
                    exist = [s for s in ds_subjs \
                     if os.path.exists(f"{scores_path}/{prefix}-{s}-{mode}.npy")]

                    if len(exist) ==0: continue;
                    is_full = len(ds_subjs)==len(exist);
                    tstat = "" if train=="pretr" else "untr_";

                    
                    is_trained = train=="pretr";

                    outp = "-" + colored(f"{tstat}{modeln}", (("green" if is_full else "blue")if is_trained else "light_grey"))
                    outp = outp+ ", " + colored(f"{mode}", ("yellow" if train=="pretr" else "light_grey"))
                    
                    if not is_full:
                        outp = outp + ": "+  colored(f"{exist}", "light_grey");
                    print(outp)
                    
        print("")
    
                