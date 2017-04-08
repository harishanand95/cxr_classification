### This is attempt at classifying CXR images into Tuberculosis and non-tuberculosis categories

China Set - The Shenzhen set - Chest X-ray Database
Description: The standard digital image database for Tuberculosis is created by the National Library of Medicine, Maryland, USA in collaboration with Shenzhen No.3 People’s Hospital, Guangdong Medical College, Shenzhen, China. The Chest X-rays are from out-patient clinics, and were captured as part of the daily routine using Philips DR Digital Diagnose systems. 
Number of X-rays: 
* 336 cases with manifestation of tuberculosis, and 
* 326 normal cases.

It is requested that publications resulting from the use of this data attribute the source (National Library of Medicine, National Institutes of Health, Bethesda, MD, USA and Shenzhen No.3 People’s Hospital, Guangdong Medical College, Shenzhen, China) and cite the following publications:  
* Jaeger S, Karargyris A, Candemir S, Folio L, Siegelman J, Callaghan F, Xue Z, Palaniappan K, Singh RK, Antani S, Thoma G, Wang YX, Lu PX, McDonald CJ.  Automatic tuberculosis screening using chest radiographs. IEEE Trans Med Imaging. 2014 Feb;33(2):233-45. doi: 10.1109/TMI.2013.2284099. PMID: 24108713
* Candemir S, Jaeger S, Palaniappan K, Musco JP, Singh RK, Xue Z, Karargyris A, Antani S, Thoma G, McDonald CJ. Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration. IEEE Trans Med Imaging. 2014 Feb;33(2):577-90. doi: 10.1109/TMI.2013.2290491. PMID: 24239990


Here is the image of an attempted small model.

* Input is in 640x480 image in greyscale.
* Split across as (182,180,300) for validation, test and train images respectively.
* (640x480x1) input after conv1 operation of stride 2 and 16 layer depth changes to (320x240x16).
* (320x240x16) input after conv1 operation of stride 2 and 16 layer depth changes to (160x120x16).
* Relu and L2 regularisation is used here to avoid traps.
* A fully connected layer follows it and flattens it to 64 neurons.
* Dropout is used as normalisation after previous layer.
* A final FC layer with softmax function outputs a score on 2 values (TB,NoTB).
* Adam optimizer is used as SGD optimiser had difficulty generating graphs. This also means more parameters to store.

![alt text](https://github.com/harishanand95/cxr_classification/blob/master/tflearn_model.png?raw=true "Model")

classification_cxr.py has a model made in tensorflow itself while tflearn_classify has tflearn library used.
