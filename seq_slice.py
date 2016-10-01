import sys, os, random
sys.path.insert(0, os.path.expanduser("~/.local/lib/python2.7/site-packages/"))
sys.path.insert(0, '/tmp3/first350/caffe/' + 'python/')
import SimpleITK as sitk
import caffe
import lmdb
import time
import numpy as np
print(np.version.version)
import numba
import Image

a = []
gt = {}
aa = {}
path="./BRATS2015_Training-2/HGG/"
path2="./BRATS2015_Training-2/LGG/"
dir_list = os.listdir(path)
dir_list1 = os.listdir(path2)
all_list = dir_list + dir_list1
#dir_list += os.listdir(path2)
def writeImage(image, filename):
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
def getFilePath(dl, path):
    for p in dl:
        if ".DS" in p:
            continue
        files = os.listdir(path+p)
        aa[p] = []
        for f in files:
            fpath = path+p+"/"+f
            if ".DS" not in f:
                ffiles = os.listdir(fpath)
            else:
                continue
            for ff in ffiles:
                if ".mha" in ff:
                    if "OT" in ff:
                        gt[p] = fpath+"/"+ff
                    else:
                    #extract_path(p,f,fpath+"/"+ff)
                        a.append(fpath+"/"+ff)
                        aa[p].append(fpath+"/"+ff)

def getMask(img):
    mask = sitk.Image(240, 240, 155, sitk.sitkInt8)
    for z in range(img.GetSize()[2]):
        for i in range(img.GetSize()[0]):
            for j in range(img.GetSize()[1]):
                if img[i,j,z] != 0:
                    mask[i,j,z] = True
                else:
                    mask[i,j,z] = False
    return mask
@numba.jit
def fastgen(images, normed_images, GT, GTMASK):
    image = images[0]
    skip1 = -1
    skip2 = -1
    blanket = range(5)
    blob = 1
    label = 1
    for z in range(image.shape[2]):
        # get position
        Img = GT[:,:,z]
        maskImg = GTMASK[:,:,z]
        for i in range(Img.shape[0]):
            for j in range(Img.shape[1]):
                if Img[i][j] == 0 and maskImg[i][j] != 1:
                    Img[i][j] = 5
        #writeImage(Img, "ColorLabel.png")
        #writeImage(maskImg, "MaskLabel.png")
        #exit(1)
        container = range(4)
        for index in range(4):
            container[index] = normed_images[index][:,:,z]
        container = np.asarray(container)
        if isinstance(blob, int):
            blob = [container]
            label = [Img[np.newaxis,...]]
        else:
            blob.append(container)
            label.append(Img[np.newaxis, ...])
    print(len(blob), blob[0].shape)
    return blob, label
def genpatch():
    blob = []
    label = []
    count = 0
    for brain in all_list:
        print(count)
        if ".DS" in brain:
            continue
        #if count  == 30:
        #    break
        #else:
        count += 1
        GT = sitk.ReadImage(gt[brain])
        GT_array = sitk.GetArrayFromImage(GT)
        GT_array = np.transpose(GT_array, (2,1,0))
        # depth, height, width (155,240,240) (z,y,x)
        images = []
        normed_images = []
        maskImage = []
        for filename in sorted(aa[brain]):
            if "T1" in filename or "T1c" in filename:
                if "N4" in filename:
                    continue
            print(count, filename)
            im = sitk.ReadImage(filename)
            #maskImage = sitk.OtsuThreshold( im, 0, 1, 200 )
            maskImage = im > 0
            maskImage = sitk.GetArrayFromImage(maskImage)
            maskImage = np.transpose(maskImage, (2,1,0))
            # aaaa = sitk.GetArrayFromImage(maskImage[:,:,100])
            # print(aaaa.flatten())

            #print(sitk.GetArrayFromImage(maskImage[:,:,100]))
            #sitk.WriteImage(maskImage[:,:,100],"Mask"+str(len(images))+".mha")
            im_array = sitk.GetArrayFromImage(im)
            im_norm = sitk.ReadImage(filename)
            im_norm = sitk.Cast(im_norm, sitk.sitkFloat32)
            normalizeFilter = sitk.NormalizeImageFilter()
            im_norm = normalizeFilter.Execute(im_norm)
            im_norm_array = sitk.GetArrayFromImage(im_norm)
            images.append(np.transpose(im_array, (2,1,0)))
            normed_images.append(np.transpose(im_norm_array, (2,1,0)))

        start = time.time()
        a, b = fastgen(images, normed_images, GT_array, maskImage)
        blob = blob + a
        label = label + b
        end = time.time()
        print("elapse time: ", end-start)
        print(len(blob))
    return blob, label

def readDatum():
    env = lmdb.open('data_lmdb', readonly=True)
    datum = caffe.proto.caffe_pb2.Datum()
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
            print(key, datum.label)
if __name__ == "__main__":
    #readDatum()
    #exit(1)
    getFilePath(dir_list, path)
    #getFilePath(dir_list1, path2)

    final_data, final_label = genpatch()
    Y =  np.array(final_label)
    print(Y[0].shape)
    print(len(Y))
    Y =  np.array(Y, dtype=np.uint8)
    #shuffle
    N = range(len(final_data))
    random.shuffle(N)
    train_len = int(len(final_data) * 0.75)
    val_len = len(final_data) - train_len
    print(train_len, val_len)
    print 'Creating image dataset...'
    X = np.array(final_data)
    #X = np.array(X,dtype=np.uint8)
    X = np.array(X,dtype=np.float32)
    
    """
    map_size = X.nbytes*10
    env = lmdb.open('data_seq_lmdb', map_size=map_size)
    env_val = lmdb.open('data_seq_val_lmdb', map_size=map_size)
    env_label = lmdb.open('label_seq_lmdb', map_size=map_size)
    env_val_label = lmdb.open('label_seq_val_lmdb', map_size=map_size)
    for index,z in enumerate(N):
        # i = N[z]
        i = index
        datum = caffe.io.array_to_datum(X[i].astype(float))
        datum_label = caffe.io.array_to_datum(Y[i])
        #datum.data = X[i].tobytes()
        if index > train_len:
            str_id = '{:08}'.format(index)
            with env_val.begin(write=True) as txn:
                txn.put(str_id, datum.SerializeToString())
            with env_val_label.begin(write=True) as txn:
                txn.put(str_id, datum_label.SerializeToString())
        else:
            str_id = '{:08}'.format(index)
            with env.begin(write=True) as txn:
                txn.put(str_id, datum.SerializeToString())
            with env_label.begin(write=True) as txn:
                txn.put(str_id, datum_label.SerializeToString())
    print('done...train')
    print("Creating Label...")
    """
