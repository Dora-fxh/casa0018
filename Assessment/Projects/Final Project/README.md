# Project purpose
Our project aims to automatically identify whether faces detected are with or without masks. It could be applied in some public places and signal the guards when finding someone who does not abide by the regulations. In this case, guards do not need to patrol around or staring surveillance cameras all the time. A lot of work for guards can be saved and the effectiveness is also improved. 
# About data
The main data set we used is collected by AIZOO and can be downloaded [here](https://drive.google.com/file/d/1QspxOJMDf_rAWVV7AU_Nc0rjo1_EPEDW/view).
It contains two folders, one is for training which has 6120 images (3006 of them come from the MAFA dataset with face masks and 3114 from WIDER Face without masks), the other is as the validation set and has 1839 pictures(contains 799 mask samples and 1040 no-mask samples). Although the dataset with mask and with no masks was mixed together, every image has a corresponding XML file which includes the label, the bounding box of the face. Thus, we use these information in the XML file to process the data (crop only the face) and divide them into mask and no mask set. The processing part is included in the **faceMask.ipynb** and the processed data is uploaded which is the **base.zip**. Besides, the data for testing is also uploaded, which is in the **test-set** folder get from [this github](https://github.com/chandrikadeb7/Face-Mask-Detection).
# About python file
There are two python files, one is the **previousModel.ipynb**, the other is **faceMask.ipynb**. 'previousModel.ipynb' contains many 'wrong' explorition we did and the model trained there uses the data that is not diverse, so the overfitting problem exists. The **faceMask.ipynb** is the final version, it uses the new dataset and some other methods were employed(For example, we randomly rescale, flip, shear, zoom, reotate and shift the photo to produce more samples to adapt the changeable environment). Thus, the generalization ability is hugely improved and has lesser overfitting problem. previousModel.ipynb can not be run by restrat kernal and run all, since some parts uses tensorflow 2.X, like in producing the frozen_graph.pd (we build and switch to a new conda environment to run this part). Thus, this file is mainly used to record how different model is saved like tflite and make the comparison with the model built by faceMask.ipynb.
# About models folder
It contains some types of the models we saved. The model we finally produced and used to test the new data and do the mask detedction using webcam is not here since it exceed the limit of github(more than 100M), but can be downloaded [here](https://www.dropbox.com/h?preview=maskmodel.h5)

# Prerequisites

The final model built in faceMask.ipynb uses the tensorflow 2.4.1 and other packages' version are listed below to refer.
Package                Version
---------------------- -------------------
absl-py                0.12.0
appdirs                1.4.4
astunparse             1.6.3
backcall               0.2.0
brotlipy               0.7.0
cachetools             4.2.1
certifi                2020.12.5
cffi                   1.14.5
chardet                4.0.0
cloudpickle            1.6.0
colorama               0.4.4
cryptography           3.4.6
cycler                 0.10.0
cytoolz                0.11.0
dask                   2021.3.0
decorator              4.4.2
flatbuffers            1.12
gast                   0.3.3
google-auth            1.28.0
google-auth-oauthlib   0.4.3
google-pasta           0.2.0
grpcio                 1.32.0
h5py                   2.10.0
idna                   2.10
imagecodecs            2021.1.28
imageio                2.9.0
ipykernel              5.5.0
ipython                7.21.0
ipython-genutils       0.2.0
jedi                   0.18.0
jupyter-client         6.1.12
jupyter-core           4.7.1
Keras                  2.4.3
Keras-Preprocessing    1.1.2
kiwisolver             1.3.1
lxml                   4.6.2
Markdown               3.3.4
matplotlib             3.3.4
networkx               2.5
numpy                  1.20.1
oauthlib               3.1.0
olefile                0.46
opencv-python          4.5.1.48
opt-einsum             3.3.0
packaging              20.9
parso                  0.8.1
pdfkit                 0.6.1
pickleshare            0.7.5
Pillow                 8.1.2
pip                    21.0.1
pooch                  1.3.0
prompt-toolkit         3.0.17
protobuf               3.15.6
pyasn1                 0.4.8
pyasn1-modules         0.2.8
pycparser              2.20
Pygments               2.8.1
pyOpenSSL              20.0.1
pyparsing              2.4.7
PySocks                1.7.1
python-dateutil        2.8.1
PyWavelets             1.1.1
pywin32                300
PyYAML                 5.4.1
pyzmq                  22.0.3
requests               2.25.1
requests-oauthlib      1.3.0
rsa                    4.7.2
scikit-image           0.18.1
scipy                  1.6.1
setuptools             52.0.0.post20210125
six                    1.15.0
tensorboard            2.4.1
tensorboard-plugin-wit 1.8.0
tensorflow             2.4.1
tensorflow-estimator   2.4.0
termcolor              1.1.0
tifffile               2021.3.17
toolz                  0.11.1
torch                  1.8.0
torchvision            0.9.0
tornado                6.1
tqdm                   4.59.0
traitlets              5.0.5
Transform              0.0.1
typing-extensions      3.7.4.3
urllib3                1.26.4
utils                  1.0.1
wcwidth                0.2.5
Werkzeug               1.0.1
wheel                  0.36.2
win-inet-pton          1.1.0
wincertstore           0.2
wrapt                  1.12.1
