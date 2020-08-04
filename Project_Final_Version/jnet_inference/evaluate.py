import os
import torch

from torch.utils.data import DataLoader
from nets import Criterion,batch_images_labels
from dataset import is_torch_none
from imageio import imwrite as imsave
from PIL import Image
import numpy as np
from skimage import img_as_ubyte

def evaluate(model,dataset,params,save_images=True,print_output=True):
  dataloader=DataLoader(dataset, batch_size=1,num_workers=params.num_workers,shuffle=False)
  model.eval()
  use_cuda=torch.cuda.is_available() and params.cuda
  model.cuda() if use_cuda else model.cpu()
  with_labels=False
  criterion=Criterion(params.dt_bound)
  err=np.zeros(3)
  return_masks = []
  if len(dataloader)>0:
    for i,batch in enumerate(dataloader):
      image,labels=batch_images_labels(batch,use_cuda)
      with_labels=not is_torch_none(labels)
      outputs=model(image)
      # if with_labels:
      #   loss,l1_seg,l1_dist=criterion(outputs,labels)
      #   err+=np.array((loss.data[0],l1_seg,l1_dist))/len(dataloader)
      #   if print_output:
      #     print("Image "+str(i)+": loss="+str(loss.data[0])+", l1_seg="+str(l1_seg)+", l1_dist="+str(l1_dist))
      # elif print_output:
      #   print(dataset.get_filename_basis(i),end=', ',flush=True)
      vis=outputs.cpu().data.squeeze().numpy()

      if save_images:
        seg=vis[0,:,:]
        dist=vis[1,:,:]
        dist[dist>params.dt_bound]=params.dt_bound
        dist/=params.dt_bound 
        return_masks.append(img_as_ubyte(seg))
        # imsave(os.path.join(params.output_dir,"seg_"+dataset.get_filename_basis(i)+".tif"),img_as_ubyte(seg))
        # imsave(os.path.join(params.output_dir,"dist_"+dataset.get_filename_basis(i)+".tif"),img_as_ubyte(dist))
        
  # if print_output:
  #   if with_labels:
  #     print("Average: "+str(err))
  #   else:
  #     print("")
  return(return_masks)

