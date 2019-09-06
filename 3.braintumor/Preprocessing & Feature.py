

from preprocessing import Preprocessing
from segmentaion import Segmentation
from matplotlib import pyplot as plt
from skimage import io

import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from skimage.color import rgb2gray




# you can make a loop to handling all images at once 
preprocessing = Preprocessing()
preprocessing.preproces('C:\\iiitnewbrain\\braintumor\\Cl\\ab5.jpg')
preprocessing.binarization()
preprocessing.removingSkul()
preprocessing.enhanceImage()
preprocessing.segmentation()
image = preprocessing.getInfectedRegion()


# In[ ]:


# read and show image 
im = io.imread('C:\\iiitnewbrain\\braintumor\\tmp\\tumourImage.jpg')
plt.imshow(im, 'gray')


# ### Extract GLCM Texture  Features

# In[ ]:


# GLCM Texture Features
ds = []
cr = []
cn = []
am = []
en = []
ho = []

glcm = greycomatrix(im, [5], [0], symmetric=True, normed=True)
ds.append(greycoprops(glcm, 'dissimilarity')[0,0])
cr.append(greycoprops(glcm, 'correlation')[0,0])
cn.append(greycoprops(glcm, 'contrast')[0,0])
am.append(greycoprops(glcm, 'ASM')[0,0])
en.append(greycoprops(glcm, 'energy')[0,0])
ho.append(greycoprops(glcm, 'homogeneity')[0,0])
    
    
print('dissimilarity',ds)
print('correlation',cr)
print('contrast',cn)
print('ASM',am)
print('energy',en)
print('homogeneity',ho)






