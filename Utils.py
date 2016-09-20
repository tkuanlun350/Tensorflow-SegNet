import tensorflow as tf
import Image
import numpy as np

def writeImage(image, filename):
    """ store label data to colored image """
    Sky = [128,128,128]
    Building = [128,0,0]
    Pole = [192,192,128]
    Road_marking = [255,69,0]
    Road = [128,64,128]
    Pavement = [60,40,222]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Sky, Building, Pole, Road, Road_marking, Pavement])
    for l in range(0,6):
        r[image==l] = label_colours[l,0]
        g[image==l] = label_colours[l,1]
        b[image==l] = label_colours[l,2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)

def storeImageQueue(data, labels, step):
  """ data and labels are all numpy arrays """
  for i in range(BATCH_SIZE):
    index = 0
    im = data[i]
    la = labels[i]
    im = Image.fromarray(np.uint8(im))
    im.save("batch_im_s%d_%d.png"%(step,i))
    writeImage(np.reshape(la,(360,480)), "batch_la_s%d_%d.png"%(step,i))
