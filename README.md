# Automatic Imaging Artifact Masking
Data cleaning via automated artifact masking in high-throughput microscopy imaging.

This is an **adaptation**  of the Score-CAM-U-Net artifact masking pipeline which is described in, but not currently made available by, the paper: **ArtSeg-Artifact segmentation and removal in brightfield cell microscopy images without manual pixel-level annotations (2022)** [1]. 

This pipeline makes use of Score-CAM [2], a technique for improving interpretability of deep-learning image classifiers, by producing visual explanations of the features relevant to model predictions. When applied to a trained classification model, Score-CAM extracts the learned features (activation maps) from the final convolutional layer, projecting each onto a separate copy of the original input image to highlight that particular feature. These projected images are then fed as separate inputs to the trained model, with the resulting target-class probabilities acting as scores of each features importance. Linear combination of each features activation map and their respective scores is then performed to create a final activation map, which is passed through a ReLU activation function to retain only the features with a positive influence on prediction of the target class. Accordingly, the final output from Score-CAM for a provided image is a saliency map highlighting image regions most influential for prediction of the target class.

The implementation of Score-CAM used in my pipeline is based on that provided by S. Tabayashi at https://github.com/tabayashi0117/Score-CAM.

For the artifact-segmentation pipeline described in Ali et al. (2022), a CNN model is trained to classify image data from cell microscopy imaging as Clean or Artifact-Containing, to which the Score-CAM framework is applied to produce saliency maps. These maps provide weakly-supervised labelling of artifacts in the image data, which is subsequently used to train a U-Net model [3] to produce more precise artifact segmentation masks for each image. U-Net is a specialised CNN architecture comprised of an encoder network for feature learning and extraction, followed by a decoder network for generation of a segmentation mask through pixel-wise classification. This mask can then be overlayed on the original data to remove the identified artifacts.

In deviation from the original Score-CAM-U-Net pipeline from Ali et al. (2022), the final saliency map produced by Score-CAM for each image is further fine-tuned in this implementation using an edge map. To generate the edge map, horizontal ($`Gx`$) and vertical image gradients ($`Gy`$) are computed on the original input image and then combined to give the gradient magnitude, using:
$`G= √(G_x^2+G_y^2 )`$. The gradient magnitude, $`G`$, is dilated and normalised between 0 and 1 to produce the edge map. The edge map for an image is then multiplied against its corresponding Score-CAM saliency map, producing a more defined outline of the predicted artifact locations. These fine-tuned saliency maps are then binarized to produce the artifact label masks for training of the U-Net segmentation model.

**[1]** Ali, M.A.S., Hollo, K., Laasfeld, T. et al. ArtSeg—Artifact segmentation and removal in brightfield cell microscopy images without manual pixel-level annotations. Sci Rep 12, 11404 (2022). https://doi.org/10.1038/s41598-022-14703-y

**[2]** Wang, H., Wang, Z., Du, M., Yang, F., Zhang, Z., Ding, S., ... & Hu, X. Score-CAM: Score-weighted visual explanations for convolutional neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops (pp. 24-25), (2020).

**[3]** Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18 (pp. 234-241). Springer International Publishing.

# Usage
To use this pipeline, a tutorial notebook for training of the artifact classification model, based on our application case described below, is provided in (ResNet50_Artifact_Classifier.ipynb). The Score-CAM.ipynb notebook then steps through generation of artifact labels of the images with Score-CAM, which are used to train the artifact segmentation UNET model in the Artifact_Segmentation_UNET.ipynb notebook. The weights for these trained models from our own application are also provided in the weights folder and can be used to initialise these models for re-training on new datasets.

A notebook tutorial for usage of the trained models for artifact segmentation of a toy dataset is then provided in Toy_data_example.ipynb.

# Application Example
This implementation was produced to enable masking of widespread artifacts in a large-scale brightfield microscopy dataset of SA-β-Gal stained Mesenchymal Stem Cells (MSCs) to allow for accurate staining quantification. As SA-β-Gal staining intensity presents as dark pixels on a light background when imaged with brightfield microscopy, removal of the variety of artifacts present in these images was essential for performing accurate quantification. SA-β-Gal levels in each cell were then used to automate per-cell labelling of cellular senescence in corresponding nuclei imaging, for training of an image-based senescence classification model. Details and code for the segmentation, automated labelling pipeline and classification model are provided in my Senescence Classifier repository.

The SA-β-Gal stained MSCs were imaged in brightfield with the Operetta system using the 40x air objective to produce a total of 13,860 images. Artifacts of some form, including general debris, blurry objects and SA-β-Gal crystals, were present across 96.3% of the total dataset according to the trained artifact classifier.

![Artifacts](imgs/Figure_3.png)

![Framework](imgs/Figure_4.png)

For this implementation of the Score-CAM-U-Net framework, I used the ResNet50 CNN architecture to train a model on a subset of SA-β -Gal images (n = 2,024) for classification as Clean or Artifact-Containing (ResNet50_Artifact_Classifier.ipynb). When evaluated on the held-out test data (n = 403), the trained model demonstrated strong performance for both Clean (F1 score = 97%) and Artifact-Containing (F1 score = 98%) classes. 

Images correctly classified as containing artifacts by this model (1,300 of the 1,327 in the dataset) were then fed to the Score-CAM framework to perform weakly-supervised pixel-wise labelling of the artifacts, which were further refined using an edge-map as described above (Score-CAM.ipynb). These fine-tuned saliency maps were then binarized to produce the artifact label masks for training of the U-Net segmentation model (Artifact_Segmentation_UNET.ipynb). 

The U-Net model trained on the Score-CAM artifact labels was able to identify and segment artifacts in the data reliably, achieving an overall pixel-wise accuracy of 99.2% and Intersection Over Union  (IOU) of 81.2% on the held-out test data (_n_ pixels = 87,920,640). As expected, performance for the Artifact-Containing class (F1 = 78%, Recall = 75% , precision= 80%) is lower than for the Clean class (all metrics = 100%), which dominates the dataset.

The segmentation masks produced by the trained U-Net model were then applied to the corresponding (illumination-corrected) SA-β -Gal -stained images, with pixel values of masked regions replaced with the median value of the SA-β -Gal image. After masking of image artifacts in the SA-β -Gal stained images, thresholding was performed to produce a binary image of SA-β -Gal staining intensity for quantification. See the Senescence_Classifier Repo for further information regarding the automated labelling pipeline & development of senescence classification models.

