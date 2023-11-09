# Automatic_Imaging_Artifact_Masking
Data cleaning via automated artifact masking in high-throughput microscopy imaging.

This is an _**adapted implementation**_  of the Score-CAM-U-Net artifact masking pipeline which is described in, but not currently made available by, the paper: **ArtSeg-Artifact segmentation and removal in brightfield cell microscopy images without manual pixel-level annotations (2022)** [1]. 

This pipeline makes use of Score-CAM [2], a technique for improving interpretability of deep-learning image classifiers, by producing visual explanations of the features relevant to model predictions. When applied to a trained classification model, Score-CAM extracts the learned features (activation maps) from the final convolutional layer, projecting each onto a separate copy of the original input image to highlight that particular feature. These projected images are then fed as separate inputs to the trained model, with the resulting target-class probabilities acting as scores of each features importance. Linear combination of each features activation map and their respective scores is then performed to create a final activation map, which is passed through a ReLU activation function to retain only the features with a positive influence on prediction of the target class. Accordingly, the final output from Score-CAM for a provided image is a saliency map highlighting image regions most influential for prediction of the target class.

The implementation of Score-CAM used in my pipeline is based on _______ available at ___________.

For the artifact-segmentation pipeline described in Ali _et al._ (2022), a CNN model is trained to classify image data from cell microscopy imaging as Clean or Artifact-Containing, to which the Score-CAM framework is applied to produce saliency maps. These maps provide weakly-supervised labelling of artifacts in the image data, which is subsequently used to train a U-Net model [3] to produce more precise artifact segmentation masks for each image. U-Net is a specialised CNN architecture comprised of an encoder network for feature learning and extraction, followed by a decoder network for generation of a segmentation mask through pixel-wise classification. This mask can then be overlayed on the original data to remove the identified artifacts.

The implementation of UNET used in my pipeline is based on _______ available at ___________.

In deviation from the original Score-CAM-U-Net pipeline from Ali _et al._ (2022), the final saliency map produced by Score-CAM for each image was further fine-tuned in this implementation using an edge map. To generate the edge map, horizontal (Gx) and vertical image gradients (Gy) were computed on the original input image and then combined to give the gradient magnitude, using:
G= √(G_x^2+G_y^2 ). The gradient magnitude, G, was dilated and normalised between 0 and 1 to produce the edge map. The edge map for an image was then multiplied against its corresponding Score-CAM saliency map, producing a more defined outline of the predicted artifact locations. These fine-tuned saliency maps were then binarized to produce the artifact label masks for training of the U-Net segmentation model. This idea was based on ___________ [4].

**[1]** Ali, M.A.S., Hollo, K., Laasfeld, T. et al. ArtSeg—Artifact segmentation and removal in brightfield cell microscopy images without manual pixel-level annotations. Sci Rep 12, 11404 (2022). https://doi.org/10.1038/s41598-022-14703-y

**[2]** Wang, H., Wang, Z., Du, M., Yang, F., Zhang, Z., Ding, S., ... & Hu, X. Score-CAM: Score-weighted visual explanations for convolutional neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops (pp. 24-25), (2020).

**[3]** Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18 (pp. 234-241). Springer International Publishing.

**[4]**

# Application Example
This implementation was produced to enable masking of widespread artifacts in a large-scale brightfield microscopy dataset of SA-β-Gal stained Mesenchymal Stem Cells (MSCs) to allow for accurate staining quantification. As SA-β-Gal staining intensity presents as dark pixels on a light background when imaged with brightfield microscopy, removal of the variety of artifacts present in these images was essential for performing accurate quantification. SA-β-Gal levels in each cell were then used to automate per-cell labelling of cellular senescence in corresponding nuclei imaging, for training of an image-based senescence classification model. Details and code for the segmentation, automated labelling pipeline and classification model are provided in _____.

The SA-β-Gal stained MSCs were imaged in brightfield with the Operetta system using the 40x air objective to produce a total of X images. Artifacts of some form, including general debris, blurry objects and SA-β-Gal crystals, were present across 96.3% of the total dataset.

![image](https://github.com/Ebony-Watson/Automatic_Imaging_Artifact_Masking/assets/52723545/d3f879e2-e70f-4d0c-bdee-330add47eb11)


![image](https://github.com/Ebony-Watson/Automatic_Imaging_Artifact_Masking/assets/52723545/09482d53-d84b-4fb0-a5a4-0cc54dc85738)

For this implementation of the Score-CAM-U-Net framework, I used the ResNet50 CNN architecture to train a model on a subset of SA-β -Gal images (n = 1,650) for classification as Clean or Artifact-Containing. When evaluated on the held-out test data (n = 405), the trained model demonstrated strong performance for both Clean (F1 score = 97%) and Artifact-Containing (F1 score = 99%) classes. 

The ResNet-50 architecture is first described in ____ and the implementation here is based on _____.

Artifact-Containing images from the SA-β -Gal data subset (n = 1,329) were then fed to the Score-CAM framework to perform weakly-supervised pixel-wise labelling of the artifacts, which were further refined using an edge-map as described above. These fine-tuned saliency maps were then binarized to produce the artifact label masks for training of the U-Net segmentation model. 

The U-Net model trained on the Score-CAM artifact labels was able to identify and segment artifacts in the data reliably, achieving an overall pixel-wise accuracy of 99.4% and Intersection Over Union  (IOU) of 80% on the held-out test data (n pixels = 87,920,640). As expected, performance for the Artifact-Containing class (F1 = 77%, Recall = 81% , precisioG= 73%) is lower than for the Clean class (all metrics = 100%), which dominates the dataset. When predicted masks were evaluated against a set of manually-annotated artifact masks (n = 100) performance remained relatively strong, obtaining a median F1, Recall, Precision, and IOU of 69.87%,76.9%, 67.61% and 53.7% respectively across all images. Notably, evaluating these results in relation to the results provided in Ali et al. (2022) finds that thid implementation of the Score-CAM framework exceeds the performance achieved by the original framework on their 3 datasets. Whilst definitive conclusions cannot be drawn without comparing performance of these pipelines on the same datasets, this result does suggest that the use of an edge-mask to refine the Score-CAM saliency maps may improve performance of artifact segmentation.

The segmentation masks produced by the trained U-Net model were then applied to the corresponding (illumination-corrected) SA-β -Gal -stained images, with pixel values of masked regions replaced with the median value of the SA-β -Gal image. After masking of image artifacts in the SA-β -Gal stained images, thresholding was performed to produce a binary image of SA-β -Gal staining intensity for quantification. See ___ and ____ for further information regarding the automated labelling pipeline & development of senescence classification models.

