# Deep Learning based Signature Detection and Verification

  
Signature verification systems are an essential part of most business practices. A significant amount of time and skillful resources could be saved by automating this process. This project demonstrates the implementation of an end-to-end signature verification system.

From the document the user selected, the signatures are extracted using YOLOv5. In real-world documents, there would be noise artifacts such as printed text, stamps etc which might seriously affect the performance of signature verification task. Thus a CycleGAN based noise cleaning method is added to tackle this. The cleaned signature is verified using a VGG16 based feature extractor, similar to Siamese Networks.  

This project is based on these two papers [[1]](https://repositum.tuwien.at/bitstream/20.500.12708/16962/1/Hauri%20Marcel%20Rene%20-%202021%20-%20Detecting%20Signatures%20in%20scanned%20document%20images.pdf) and [[2]](https://arxiv.org/abs/2004.12104).  
[[1]](https://repositum.tuwien.at/bitstream/20.500.12708/16962/1/Hauri%20Marcel%20Rene%20-%202021%20-%20Detecting%20Signatures%20in%20scanned%20document%20images.pdf) studies the usage of different object detection algorithms for signature detection and the results indicate that YOLOv5 outperforms all other models for the signature detection task. [[2]](https://arxiv.org/abs/2004.12104) provides a CycleGAN based approach to clean noise artifacts from signatures that are present in real-world documents and methods to perform signature validation using Representation learning.  

This project has been trained and tested on signature datasets ([Tobacco 800](http://tc11.cvc.uab.es/datasets/Tobacco800_1) and [Kaggle Signature Dataset](https://www.kaggle.com/robinreni/signature-verification-dataset)).
  

**Model weights and data is not added, will update soon. :)**  
  
## Workflow

The project works in three phases.
![Pipeline](Images/pipeline.png)

### [Signature Detection](Training/YOLOv5/)
![DetectionExample](Images/detection_from_document.jpg)  
Once the document to run inference is selected by the user, a [YOLOv5](https://github.com/ultralytics/yolov5) model will be run to detect and crop the signatures present in the document. YOLO model is trained using custom dataset created from [Tobacco 800](http://tc11.cvc.uab.es/datasets/Tobacco800_1) dataset. The notebook to convert Tobacco 800 dataset to YOLOv5 format could be found [here](Training/YOLOv5/Converting_Tobacco800_Dataset_to_YOLOv5_Format.ipynb)

### [Signature Cleaning](Training/CycleGAN/) 
!['Gan Example Real'](Images/cleaning.jpg)  
Signatures on real-world documents often contains noise artifacts like stamps/seals, text and printed lines. These noise artifacts might affect the signature verification process. A noise cleaning method based on [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) will be performed on the detected signatures to generate noise free signatures. The CycleGAN model is trained using Kaggle Signature Dataset. Noisy signatures are generated from the dataset using OpenCV.[This notebook](Training/CycleGAN/CycleGAN-Data_Preperation_Adding_Noise.ipynb) contains code to generate noisy images and to convert the dataset to CycleGAN input format.

### [Signature Verification](<Training/VGG16 FE/>)
![vgg_model_working](Images/verify.png)  
In the final phase, a VGG16 based feature extractor is used. The model is fine-tuned on the Kaggle Signature dataset to learn the writer independent signature representations, thus new user signatures can be added to the system without re-training the model.  
The cleaned image from the document and the reference signature (anchor image) of the user is fed into the model. The model outputs a vector (feature) that represents the signature. The features extracted from both the anchor image and cleaned signature from the document is used to compute the cosine similarity. Cosine similarity tells us how similar these two images are and it's value ranges from 0 to 1. From my experimentation, I have found out that for a matching signature pair the values are close to 1 and for a non-matching signature pair, the values are below 0.7. So I recommend a cosine similarity score of 0.8 as a threshold value to decide whether the signatures are a match or not. A more detailed take on the thresholds could be found on the [recommendations](#recommendations) section.  

## RUN THE UI APP
Install the requirements by running `pip install -qr requirements.txt`  
To run the app, `streamlit run ui.py   `

## To Train the models with custom dataset
Each component of the app is trained individually and instructions and more information can be found on their respective pages.  
**[YOLOv5 for Signature Detection](Training/YOLOv5/)**  
**[CycleGAN for Signature Cleaning](Training/CycleGAN/)**  
**[VGG16 Feature Extractor for Signature Verification](Training/VGG16FE/)**  

## Folder Structure
`Streamlit_APP`  
&nbsp;&nbsp;&nbsp;&nbsp; |-> `SOURCE`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-> `yolo_files`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-> `gan_files`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-> `vgg_finetuned_model`   
&nbsp;&nbsp;&nbsp;&nbsp; |-> `media`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-> `documents`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-> `UserSignaturesSquare`  
&nbsp;&nbsp;&nbsp;&nbsp; |-> `helper_fns`   
&nbsp;&nbsp;&nbsp;&nbsp; |-> `results`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-> `yolov5`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-> `gan`  

### SOURCE
Contains the trained models and codes for YOLOv5, CycleGAN and VGG16 feature extractor.  

#### yolo_files
`detect()` function in `detect.py` is used to perform signature detection. Inside `detect()` function, a dictionary `opt` is initialized to set some parameters.  
> **Some important parameters**  
`'weights': 'SOURCE/yolo_files/best.pt'` - Model weight file path.  
`'source': image_path` - Path to document images to run detection.  
`'classes': 1` - Model is trained on Logo (0) and Signature(1) classes. classes=1 force the model to detect only signatures.  
`'project': 'results/yolov5/'` - Path to save the detected images.  

#### gan_files
Model weights are stored in `checkpoints\gan_signdata_kaggle`. `clean()` function in `test.py` is used to perform signature cleaning.  

Some important parameters are added as default arguments in `base_options.py` and `test_options.py` files in `options` folder.  
> In `base_options.py`  
`dataroot', default='results/gan/gan_signdata_kaggle/gan_ips/testB'` - Path to input images (detections from YOLO)  
`name': 'gan_signdata_kaggle'` - Name of the trained model.  
`gpu_ids': '0'` - Should use -1 to perform inference on cpu  
`checkpoints_dir': 'SOURCE/gan_files/checkpoints/'` - Model weight file path.  
`model': 'test',` - test model is used for performing inference.  

> In `test_options.py`  
`results_dir: './results/gan/'` - Path to save results.  

#### vgg_finetuned_model
`verify()` from `vgg_finetuned_model\vgg_verify.py` is used to perform verification.  
VGG16 model fine-tuned on [custom Kaggle signature dataset]() is used to fine-tune the model. The output of the first fully connected layer is used to extract features. Since the VGG16 model is trained to learn writer independent signature representations, new signatures can be added to the database (media/UserSignaturesSquare) without retraining the model.  
More information regarding the logic used could be found under `ui.signature_verify()` function.  
  

**Model file for vgg16 feature extractor would be made available soon.


### media
Contains two sub-folders `documents` and `UserSignaturesSquare` which contains the document images and Reference (Anchor) Signature images respectively.  
> To convert pdf files to images **pdf2image** library could be used.
```
from pdf2image import convert_from_path  
images = convert_from_path(document_path)
```

### helper_fns
Contains `gan_utils.py`. The `resize_images()` function is used to convert a signature image to the input requirements of CycleGAN model for inference.  

### results
Stores the results of YOLOv5 and CycleGAN.  
* YOLOv5 results are stored `yolov5` folder. A new folder `exp` is created every time the model is run.   
* CycleGAN requires inputs in a particular folder structure. As CycleGANs perform image (domain A) to image (domain B) translation, it can perform noisy signature to clean signature translation and vice versa. To perform noisy (domain B) to clean (domain A) transformation, the images should be stored in a folder named `testB`.  
The cleaned signatures are stored in `results/gan/test_latest/latest/`. The real images are subscripted as `real` and generated images as `fake`.  


## Recommendations
For Signature Verification, the POC illustrates that matching signature pairs have a cosine similarity score close to 1 (0.8 to 1. from my observations) and <0.7 for non-matching pairs. Thus for cosine similarity scores greater than 0.8, we can classify it as matching/verified with high confidence. Similarly, for scores less than 0.7 can be classified as non-matching. Cosine similarities between 0.7 and 0.8 a manual verification workflow could be used.  
  
If we take a pair of non-matching signatures, the while background is common for both of the signatures. The only distinctive difference is the handwritten signature part, which covers a very minimal area. Thus the common white background leads to a score close to 0.5 or 0.7. But the strong signature representations learned by the model pushes the score of matching pairs towards 1.
