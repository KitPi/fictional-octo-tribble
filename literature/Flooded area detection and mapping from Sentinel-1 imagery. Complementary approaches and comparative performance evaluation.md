Flooded area detection and mapping from Sentinel-1 imagery. Complementary approaches and comparative performance evaluation

Andrei Toma\textsuperscript{a}, Ionuț Šandric\textsuperscript{b} and Bogdan-Andrei Mihai\textsuperscript{b}

\textsuperscript{a}Doctoral School of Geography ‘Simion Mehedinti’, Faculty of Geography, University of Bucharest, Bucharest, Romania; \textsuperscript{b}Faculty of Geography, University of Bucharest, Bucharest, Romania

**ABSTRACT**

The current study assesses the performance of several machine learning (ML) and deep learning (DL) models for detecting and mapping floods using Sentinel-1 SAR imagery. Three distinct approaches were used: pixel classification, object-based image analysis and object instance segmentation. The ML models are Random Forest, and Support Vector Machine and the DL models are U-NET, DeepLabV3 and Mask RCNN. Five different case studies were selected to test the models’ scalability. These areas are in Romania (Prut River, at the border between Romania, the Republic of Moldova and Ukraine, Timiş River, and Răul Negru River), the United States of America (Missouri River) and Australia (Broughton Creek). For each flood, five Sentinel-1 images were used, four collected before the flood and one collected after the flood. The intensity images were stacked and scaled in the range of the intensity thresholds associated with water and non-water so that all the case studies have the same margins for intensity. Samples of water, vegetation, agricultural fields, and bare soil were collected only from the Prut River case study and used in the training process. Out of all models, the U-Net model returned the highest accuracy with a value for Intersect over Union of 0.763 for a tile size of 128x128 pixels.

**Introduction**

Radar imagery is one of the most representative datasets in flood mapping, offering a more accurate depiction of water-covered areas than optical data (Schumann & Moller, 2015). Unlike optical data, radar imagery acquired by active sensors is not dependent on the external illumination from the sun and can be retrieved even under conditions of fully cloud-covered skies due to its centimeter-scale wavelengths. This makes radar imagery exceptionally valuable for natural hazard management and sustainable territorial planning, as it can operate day and night in all weather conditions (Zhang et al., 2020).

The European Space Agency’s Copernicus Sentinel-1 SAR satellite, with its medium spatial resolution of 20 m in its primary Interferometric Wide (IW) mode used over land, and six-day revisit time (when two satellites are used), is key for flooded area detection and mapping over large areas, from continental to country level (Rahman et al., 2021; Yang et al., 2021). It allows a systematic application of multi-date imagery on major river floodplains (Uddin & Meyer, 2019), as well as on the extended lake regions (Dong et al., 2023), narrow mountain valleys (Reksten et al., 2019), and rugged hilly areas (McCormack et al., 2022). Sentinel-1’s C-band radar coherence datasets, enhanced with advanced noise multilook processing, are utilized in various mapping methodologies and algorithms in the literature (Wagner et al., 2020). However, these approaches often focus on complementary mapping techniques rather than conducting comparative analyses of these methodologies across different sites with unique geographic features.

The multiday image thresholding based on histogram differences is a simple approach (Markert et al., 2020) that involves searching for the most appropriate backscatter intensity coefficient values between water-covered zones and the rest of the image (Chakma & Akter, 2021), in order to produce polygons to be used further in flooded area assessment (Moharrami et al., 2021) and future predictive scenarios (Pedzisai et al., 2023). The limited accuracies of the derived maps on more complex floodplain topography and land use (Tran et al., 2022), where other radar image features cause shadowing, made necessary an adaptation of the algorithms (Chen et al., 2021). This effect is visible after comparing large flooded areas on large river floodplains with smaller and narrower flooded areas of smaller rivers, which require higher resolution data (Markert et al., 2020). A hybrid integration of image thresholding...
with other approaches, such as machine learning (ML), can enhance the accuracies (Wang et al., 2022).

Radar textural features in flooded areas often introduce uncertainty, especially at the edges of water-covered regions (Tavus et al., 2022). Complex radar signatures emerge from water superposition on different land cover classes at different depths (Hao et al., 2021), from agriculture to forest stands and settlements as well (Carreño Conde & De Mata Muñoz, 2019). In this respect, there is a growing interest in adapting algorithms on different combinations of uncorrelated polarization layers (Landuyt et al., 2020) as well as on a complex stack of radar data and ancillary data (Colacicco et al., 2024) including optical-derived data (Kim et al., 2021).

Machine learning algorithms, such as Support Vector Machine (SVM) or Random Forest (RF) (Huang et al., 2021) are viable options for flood area mapping when the training datasets are sufficiently representative and consistent (Hardy et al., 2019). However, pixel-based supervised training and classification methods require significant expert knowledge and are often limited by the number of classes that can be accurately represented. This limitation arises because many pixels contain mixed features, such as water-covered surfaces combined with land cover or land use features. To address these challenges, an object-based image analysis (OBIA) approach can be employed (Zhang et al., 2021). OBIA focuses on segmenting the pixels (Gašparović & Klobučar, 2021) at various image resolutions to identify the most homogeneous feature clusters. These clusters are then grouped into a more accurate layer following a rigorous selection process based on specific criteria (Cao et al., 2019). This method reduces the reliance on expert knowledge and improves classification accuracy by considering spatial context and relationships.

In continuity with these types of approaches, it is more likely to train advanced algorithms on the image sets (Bonaﬁlia et al., 2020). Deep learning (DL) based image analysis with specific and adapted architectures can extract more accurately the flooded areas (Jiang et al., 2021). By selecting a collection of attention features (Jamali et al., 2024), DL models can offer a detailed view of texture-related information. This enables the detection of convolutional water-related features and the distinction from other ground elements such as buildings, agricultural plots, forest stands and transportation networks, after performing systematic image encoding and decoding stages.

Understanding the unique geographic features of flooded areas is important for accurate mapping (Drakonakis et al., 2022). The accuracy of flood maps can vary significantly depending on the parameters used in the algorithm, even for the same flood event or when mapping multiple events in the same area (Tsyganskaya et al., 2018). While the quality and consistency of training data are important for improving accuracy, results are not solely dependent on the quantity of texture samples in contextual feature patterns. Flooded areas often form highly complex clusters that require a different approach compared to well-known spatial data. These clusters can combine flooded regions with various landscape features in intricate ways, making it challenging to extract consistent and robust training data (Ulloa et al., 2022). Selecting the optimal pattern group as a sampling feature in a given geographical context is a significant challenge that must be addressed to improve mapping accuracy.

Comparison studies that evaluate the most commonly employed flood mapping and modelling techniques using radar data are relatively limited (Shen et al., 2019). These studies are essential for understanding the specific advantages and limitations of each technique. Most existing research focuses on comparing results within the same area using two groups of methods, such as thresholding versus machine learning (Antzoulatos et al., 2022; Bayık et al., 2018). Such comparisons primarily aim to identify the simplest method that achieves high accuracy, especially when processing large volumes of data. This includes the use of cloud-based big data analysis for the development and validation of specific products (Halder & Bose, 2024).

Our approach explores the SAR Sentinel-1 – GRD type image processing for accurate mapping of flooded areas. We evaluate well-known classification algorithms, including pixel-based, object-based, machine learning and deep learning techniques, in various geographical regional contexts. The primary objective is to assess the efficacy of utilizing C-band derived flood data for practical applications in regional to local scale risk management. The study areas are selected based on their topographic features represented by medium-to-large river floodplains, large confluences, and subsident plain areas. There are also differences in terms of climatic types and recurring flood events, ranging from temperate continental excess to temperate humid and semidesert regions. On these geographic foundations, a highly diverse array of land cover and land use intertwines with the backscatter intensities of the flooded areas.

Materials and methods

Case studies

Complementary study areas from different world regions were selected in order to evaluate the spatial accuracy of flooded area classification after the evaluation of multitemporal datasets for the most affected
flooding zones on some representative events. SAR has a recognized advantage over passive optical sensors in flooded area mapping (Lin et al., 2016), as there are no limitations related to time of day or weather conditions. However, challenges arise from adapting to water extent patterns, which cause significant surface roughness differences, needing a selection of representative images for the period of flooding under investigation (Klemas, 2015). In the current context of extreme weather episodes occurring due to global change, the analysis must adapt to the flash floods in densely built-up areas (Tay et al., 2020), even though the Sentinel-1 temporal resolution is limited to only six-day intervals when two satellites are in operation.

Following these aspects, our approach is based on five complementary case studies (Table 1), covering different flood events. Three of these events occurred in a temperate region with heavy summer rainfalls (Romania), one in Australia following a severe drought period (subtropical humid region) and one in the United States in a temperate humid climate with sudden snowmelt and frozen ground conditions. The latter two examples illustrate significant regional impacts with rapid and extensive flooding, while the first three case studies had more localised yet profound effects on communities and economies alike. A case study was selected for the training and testing of ML and DL algorithms for the task of flood mapping (see Table 1), while the other four were chosen exclusively to evaluate the scalability performance of the models in diverse and unfamiliar scenarios.

There are three case studies from Romania covering almost all representative flood types in these regions. First (Figure 1a) is a typical main river floodplain of the Prut River, with springs in the Eastern Carpathians of Ukraine, affected by floods after the cumulated rainfalls combined with snowmelt runoff in late spring. Second (Figure 1b) is a different situation of high discharges of water in the largest tectonical intermontane depression of the Carpathians, in a local closed-subsidence zone, where Olt River receives important water volumes from its tributaries such as Răul Negru after a short but aggressive summer rainfall episode. Third (Figure 1c) is the Banat Plain, where the main rivers have no higher banks, and only the damming and drainage of the last two centuries have made possible the use of land for building and agriculture. Here, an extreme flood is investigated after the main anti-flood protection dam failed at two points, under huge water pressure after heavy and cumulated rainfalls following a rapid snowmelt in the eastern mountain areas (Timiş Basin in the Southern Carpathians).

The other two examples (Figure 2d,e) differ significantly in spatial and temporal scales, as well as in specific topographic contexts. Both flooding events were triggered by a sudden weather condition change: a short-time spectacular de-icing of soil in the United States Interior Plain area drained by Missouri River and its tributaries, and limited soil drainage capacity during the drought season in New South Wales, Eastern Australia. Both situations are typical examples of historical floods affecting much larger areas compared to the first three case studies. Both events impacted the agriculture, infrastructure and parts of local communities, although the duration of water retreat after reaching peak levels varied (see Table 1).

Datasets

A dataset of 25 satellite radar images was compiled for this research from ESA’s Copernicus data portal. For each of the five case studies, a set of five images was selected, consisting of one post-event image and four pre-flood images, which were utilized to clearly differentiate flooded areas from the ones permanently covered by water.

Workflow

The overall workflow for detecting flooded areas primarily comprises three key stages: data collection and processing, training ML/DL models and mapping floods with ML/DL models. Figure 3 provides a flowchart of the methodology. Pixel-based and object-based image analysis was used in combination with machine learning and deep learning models.

Radar images pre-processing

Accurate orbit information is critical when stacking multi-temporal data, so the Sentinel-1 GRD VH radar images were updated with precise orbit information, providing the exact position and velocity of the satellite at the time of acquisition. In addition, inherent thermal noise was eliminated, and back-scattered signal intensity was normalised across scenes to reduce discrepancies between image strips. To allow comparisons between products from different data sets and different image geometries, the images were then radiometrically calibrated. The SRTM 30 m DEM was used to geocode the datasets. The images were geocoded, co-registered and stacked into a multi-band file, with the bands sorted by date of acquisition. The Refined Lee filter (Yommy et al., 2015) was used for multi-temporal speckle filtering due to its great edge-preserving capabilities.

Flood labelling

For all five case studies, the extent of the flood was manually digitized using ArcGIS Pro. Because the Prut River case study comprises most of the characteristics of the other four case studies, the sampling was
Table 1. Flood event characteristics of the case studies (data collected from various local, regional, water management authorities, and verified press websites.)

| River                        | Location/Date                          | Type of Site | Climate Type                                      | Topography                                      | Land Cover Affected                  | Triggering Factors                                                                 | Approximate Duration | Magnitude (m above danger level) | Mapped Area                                      |
|------------------------------|----------------------------------------|--------------|--------------------------------------------------|-------------------------------------------------|---------------------------------------|------------------------------------------------------------------------------------|----------------------|----------------------------------|--------------------------------------------------|
| Prut River                   | Romania-Moldova Border Zone, June 2020 | Training/Validation | Temperate continental, Scandinavian-Baltic influences | Floodplain cut in Moldavian-Podolic plateaus | Rural built-up, arable, woodland      | Cumulative heavy rainfalls                                                          | 3 days               | 2-3 m (June 26, 2020)            | Part of Romania-Moldova Border Zone              |
| Răul Negru                   | Curvature Carpathians, Romania, June-July 2018 | Validation   | Temperate continental, Intermontane microclimate, Eastern influences | Local subsidence plain area, intermontane depression of Brașov junction area | Rural built-up, arable, pasture, woodland, roads, railways | Cumulative heavy rainfalls                                                          | 7 days               | 0.2-1 m (July 02, 2018)          | Main junction Olt-Răul Negru-Târlung             |
| Timiş River                  | Western Romanian Plain, June 2020       | Validation   | Temperate continental, Sub-Mediterranean influences | Subsidence plain, low plain of Banat/Timiş Plain (flat area) | Rural built-up, arable, pasture, roads, railways | Protection dam failures after cumulative heavy rainfalls                           | 3 days               | +1.5 m (June 19, 2020)           | Bega-Timiş low floodplain                       |
| Broughton Creek-Shoalhaven Rivers Junction | New South Wales, Australia, February 2020 | Validation   | Humid subtropical                                | Floodplains junction, acute in Southern Tableland plateaus region | Arable, pasture, woodland, farms, urban-rural built-up, roads | Extreme heavy rainfall after long drought                                           | 7-10 days            | +1.5 m                            | Shoalhaven River confluence area                 |
| Missouri River               | United States, 2019                    | Validation   | Temperate continental, Humid                     | Large floodplain, Interior Plain region          | Arable, urban-rural built-up, roads, railways | Rapid snowmelt, heavy rainfall, frozen ground; dam failure, infrastructure damage | Floods persisted from March through December (historical event) | 1.5-2 m                          | Missouri River along Nebraska, Kansas and Missouri borders |
performed only for this region. This decision was also based on the context of validating the flood detection performance of the methods compared in the current study.

For the OBIA analysis, the segmentation process was optimised through interactive adjustments of object scale values until the flooded areas were clearly distinguished throughout the Prut River case study. Over-segmentation was preferred, so that the large flooded areas were identified from several objects.

To fine-tune the hyperparameters of the deep learning models, several chip sizes (128 pixels, 256 pixels and 512 pixels) were used when exporting the training samples. To increase its diversity and robustness, data augmentation was applied by scaling, flipping, cropping, adding noise, and rotating or adjusting the brightness and contrast of each tile.

**Machine learning methods**

Machine learning algorithms, as a subset of artificial intelligence (AI) algorithms, are widely used for efficient classification and basic clustering of data at a low computational cost, making them a powerful tool for data analysis. ML algorithms can learn from the data they are trained on, allowing them to adapt and improve their performance over time. ML algorithms serve as valuable tools across various applications, including image classification and pattern recognition. For the current case studies, the random forest and support vector machine algorithms were selected.

Random Forest, developed by (Breiman, 2001), is widely recognized for its ability to quickly and accurately classify data. It has been used for over two decades in various applications and is known for its efficiency and effectiveness in data classification tasks. It is an ensemble model, part of bagging algorithms, that uses pixel values extracted from the multiband imagery to recursively split and train simple models. The final prediction in a RF is made by combining the predictions of all trees obtained from the simple models.

The Support Vector Machine (SVM) algorithm, based on statistical research by (Cortes et al., 1995) works by transforming the input data into multi-
dimensional planes based on the number of classes present in the data, allowing to accurately label all value points resulting from the separations. While the underlying principles of the SVM algorithm are relatively simple, it is known for its strong performance in a variety of classification tasks due to its ability to effectively separate the data into distinct groups.

**Object-based image analysis**

Object-based image analysis (OBIA, also known as GEOBIA) is a commonly used method in various fields of image processing (Drăguț et al., 2006). OBIA’s purpose is to group pixels that have similar spectral and contextual information into homogeneous segments of various shapes and sizes (Cao et al., 2019). Therefore, it is useful for water delineation, as water on radar images appears to be very homogeneous due to its low backscatter values as opposed to other ground elements. By combining OBIA with ML, one can obtain results comparable to those generally obtained by deep learning models (Liu & Abd-Elrahman, 2018; Huang & Jin, 2020).

**Deep learning models**

**Mask R-CNN**

Mask R-CNN (He et al., 2018) is an object detection algorithm designed for precise instance segmentation in images. It extends the capabilities of Faster R-CNN (Ren et al., 2017) by not only detecting objects but also providing pixel-level segmentation, accurately outlining, and identifying each object within an image with their specific masks.

**U-Net**

U-Net is a convolutional neural network originally designed for applications in medical imaging, where training data samples are rather limited. U-Net has been shown to have the ability to learn and extract extensive features from a small number of training samples (Ronneberger et al., 2015). The architecture is based on a down-sampling network development, where successive convolutions are used to increase the complexity of the samples and learn as many details as possible. This is followed by an up-sampling network, where characteristics from the compression region are also used, through a bypass technique, to provide a context...
for localization. The goal of this process is to restore the map of the compressed features to the original size of the input image, thereby increasing the size of the elements.

**DeepLabV3**

DeepLabV3 (Chen et al., 2018) stands as a significant advancement in semantic segmentation models. This iteration marks the third generation of the algorithm, showcasing substantial improvements over its predecessors. Notably, it streamlines the processing by omitting the previous reliance on conditional random fields, resulting in reduced resource requirements and faster data processing. The architecture of DeepLabV3 incorporates common convolutional neural network layers for down-sampling data, allowing the algorithm to focus on specific image features. By eliminating redundant information, the algorithm reduces image resolution progressively. Then, atrous (dilated) convolutional layers are introduced, effectively expanding the size of the feature map.

**Results**

The quantitative comparisons provided in Tables 2 and 3 encapsulate the overall performance evaluation of the seven selected approaches in this study, measured by Intersection over Union (IoU) scores.
(Şandric et al., Kuala Lumpur, Malaysia 2022). Only the IoU metric was used to compare and discuss the results across all ML and DL models. The reason for choosing only one validation metric is the need to have a homogeneous comparison of the results, rather than comparing metrics specific to each category of models. Thus, IoU is considered to be the most objective validation metric.

**Flood mapping**

A total of 35 distinct classification results were obtained, having seven model results for each case study (Figure 4).

Table 2 and Tables 3 show a comparative evaluation of the ML and DL models, which were applied on all five case studies. These findings show significant differences in performance between the different models and case studies. The overall average performance of each method across all locations is highlighted by the mean IoU values.

For the Missouri River case study, the flooded area is large, covering 163 km². All three DL models were able to capture most of the flooded area with each model achieving an IoU above 0.75. In comparison, the ML models, performed poorly with IoU values of 0.375 for SVM and 0.527 for RF when applied on pixels, while, when applied on objects, the IoU values are comparable to the DL models (above 0.75) (Table 2 and 3).

Broughton Creek and Shoalhaven River flooded area spans over approximately 18 km². Across this region, the DL models consistently provided precise flooded area detection and mapping, with IoU values above 0.80. In contrast to the Missouri River case study, the ML models, applied both on pixels and objects, generated similar results to the DL ones, scoring IoU values above 0.80 and showing marginal differences between them (Table 2 and 3).

The flood area in the Răul Negru case study, spanning 15 km², appears fragmented, consisting of small segments interspersed with a slightly larger central part. Compared to the previous two case studies, all models (ML and DL) exhibited lower scores here. The DL models returned relatively high values, with U-Net achieving the highest score of 0.749, followed by DeepLabV3 at 0.683, and Mask R-CNN at 0.635. Conversely, the ML models performed poorly in this context, with a scoring of 0.524 for SVM and of 0.337 for RF. Better results were obtained with ML models applied on objects, with IoU values close to 0.70 (Table 2 and 3).

The Timiş River case study represents the smallest flooded area of those examined in the paper, spanning over approximately 5 km². The presence of permanent water is barely visible in the imagery, probably due to the relatively moderate resolution of the SAR imagery. Floodwater is predominantly concentrated along the river, with minimal extension into the surrounding areas. The DL models performed poorly in this region, with the highest IoU scores being 0.568 for U-Net, 0.543 for DeepLabV3 and 0.528 for Mask R-CNN. The ML models also performed poorly, with SVM and RF scoring 0.28 and 0.153 respectively. Better results were obtained with ML models applied on objects, with IoU values close to 0.60 (Table 2 and 3).

With a flooded area covering 8 km², in the case of the Prut River case study, the DL models achieved very good IoU values of 0.851 for U-Net, 0.799 for Mask R-CNN, and 0.79 for DeepLabV3. Similarly, the ML models, SVM and RF, demonstrated commendable scores of 0.782 and 0.719, respectively, when applied on pixels. The accuracy is slightly increased when the ML models are applied on objects, reaching 0.821 and 0.809 for SVM and RF respectively (Table 2 and 3).

### Table 2. Assessment of ML models’ performances on flooded area classification case studies.

| IoU scores | SVM | RF | Mean | SVM OBIA | RF OBIA | Mean |
|------------|-----|----|------|----------|---------|------|
| Missouri River | 0.375 | 0.527 | 0.451 | 0.773 | 0.775 | 0.774 |
| Broughton Creek | 0.807 | 0.795 | 0.801 | 0.862 | 0.859 | 0.861 |
| Răul Negru | 0.524 | 0.337 | 0.431 | 0.685 | 0.705 | 0.695 |
| Timiş River | 0.28 | 0.153 | 0.217 | 0.594 | 0.58 | 0.587 |
| Prut River | 0.782 | 0.719 | 0.751 | 0.821 | 0.809 | 0.815 |
| Mean | 0.554 | 0.506 | – | 0.747 | 0.746 | – |

### Table 3. Assessment of DL models’ performances on flooded area classification case studies. It summarizes the IoU scores across three DL models—Mask R-CNN, DeepLabV3, and U-Net—using different sample sizes (512 pixels, 256 pixels, 128 pixels) for each of the five case studies.

| IoU scores | 512 pixels | 256 pixels | 128 pixels |
|------------|------------|------------|------------|
| Case studies | Mask R-CNN | DeepLabV3 | U-Net | Mean | Mask R-CNN | DeepLabV3 | U-Net | Mean | Mask R-CNN | DeepLabV3 | U-Net | Mean |
| Missouri River | 0.377 | 0.64 | 0.786 | 0.601 | 0.15 | 0.057 | 0.51 | 0.239 | 0.785 | 0.765 | 0.801 | 0.784 |
| Broughton Creek | 0.663 | 0.76 | 0.809 | 0.744 | 0.746 | 0.728 | 0.799 | 0.758 | 0.807 | 0.828 | 0.847 | 0.827 |
| Răul Negru | 0.485 | 0.62 | 0.753 | 0.619 | 0.637 | 0.552 | 0.613 | 0.601 | 0.635 | 0.683 | 0.749 | 0.689 |
| Timiş River | 0.295 | 0.495 | 0.558 | 0.449 | 0.471 | 0.459 | 0.559 | 0.496 | 0.528 | 0.543 | 0.568 | 0.546 |
| Prut River | 0.739 | 0.735 | 0.856 | 0.777 | 0.75 | 0.595 | 0.845 | 0.73 | 0.799 | 0.79 | 0.851 | 0.813 |
| Mean | 0.512 | 0.65 | 0.752 | – | 0.551 | 0.478 | 0.665 | – | 0.711 | 0.722 | 0.763 | – |
Overall, the highest IoU mean value across all five case studies was achieved by U-Net when used with tile sizes of 128 by 128 pixels. On the other hand, the lowest accuracy in detecting and mapping flooded areas belongs to the RF with an IoU mean value of 0.506 for all the case studies.

Discussions

In this study, we have developed a semi-automated workflow to evaluate the efficacy of prevalent ML and DL models in mapping flooded areas using SAR data.

Spatial resolution influence on mapping floods

We have based our work on the publicly available Sentinel-1 datasets, more precisely on the GRD VH products (Ulloa et al., 2022). The 20-m spatial resolution, with 10-m pixel-spacing, proved fairly accurate for small scale flooded areas like Prut (8 km$^2$), medium-sized zones such as Râul Negru (15 km$^2$) and Broughton Creek (18 km$^2$), and larger flooded regions like Missouri (163 km$^2$). However, in the case of smaller floods, covering areas equal to or less than 5 km$^2$ and having narrow widths, as is the case of the Timiș flood, this resolution proved insufficient (McCormack et al., 2022). The Timiș flood exhibited the lowest overall IoU score among all the analysed flooded zones, 0.46, almost equal to events from France and Albania (Drakonakis et al., 2022).

On the training of machine learning and deep learning models

In case studies where the flooded area was continuously distributed without fragmentation and with backscatter homogeneity (Gašparović & Klobučar, 2021; Soria-Ruiz et al., 2022), the best flooded area detection was obtained (Figure 4). ML methods applied for pixel classification gave the lowest IoU values (Table 2). This is explained by the fact that these methods do not consider the spatial context, unlike the ML methods applied by OBIA. In the case of OBIA, the segmentation of an image into objects offers a great advantage by introducing the spatial context. All the ML classification methods used in this study, applied to objects obtained from the OBIA analysis, gave IoU values close to those obtained by the DL models (Liu & Abd-Elrahman, 2018; Huang & Jin,
The selection of the optimal values for the object segmentation is one of the crucial steps for the result and requires the use of an iterative process until the local variance is reduced to an acceptable level (Drăguț et al., 2010).

One of the main drawbacks of deep learning methods is still the large amount of training data and the time required to train the models. These issues are closely related to computing power and the cost of acquiring it. For this reason, in this article we have focused on established deep learning models that can run on computers with 8GB RAM graphics cards. We also tested the performance and transferability of the deep learning models compared to the machine learning models. We did this using training data collected only for the Prut River study area. As expected, the deep learning models gave the best results (Tables 2, 3), returning high IoU values for all study areas (Konapala et al., 2021). Training with datasets from only one case study proved to be effective, and the trained model performed similarly for all four other case studies (Figure 4). This is explained by the fact that the overall backscatter response for floods is not very different. The main difference is given by the presence of vegetation, buildings and roads (Zhao et al., 2022), agricultural land, but if these areas are covered by water, the model can detect the flooded area (Mayer et al., 2021).

**Chip size influence on deep learning models**

The results show very small differences between the three different chip sizes used in the current study (Figure 5), all close to the value of 0.7 (Yadav et al., 2022). The highest mean IoU (U-Net model) was obtained for the 128-pixel chip size, reaching a value of 0.763, compared to 0.752 for the 512-pixel chip size and 0.665 for the 256-pixel chip size (Fraccaro et al., 2022). Considering these results, we believe that all dimensions are suitable for training deep learning models for flood detection, but we recommend using the 128 pixel chip size as it requires less computational resources.

**Models’ performance vs study areas**

The advantages of ML and DL models are highlighted by their reproducibility and transferability by applying retraining with smaller training data (Hardy et al., 2019). However, the main challenge is to find and use appropriate images and labels that match as many cases as possible (Islam & Meng, 2022). Compared to ML models (SVM and RF) applied to pixels and objects, we can see better performance using OBIA extracted objects.

The U-Net model stands out by delivering superior accuracy scores across three out of five floods of
various sizes and stages of development (Figure 6). The closest scores to it are obtained by ML models (Pantazi et al., 2022) that have been applied to objects via OBIA (Cao et al., 2019). The capability to yield high accuracy without the need for supplementary processing steps underscores the efficiency and effectiveness of the U-Net model in flood analysis. While the other two DL models outperform the standalone ML models used in this paper, their performance appears relatively weaker when assessed against the ML OBIA analysis.

By comparing the results obtained by the DL models (Figure 5) trained on the large, 512-pixel sample sizes, it is evident that U-Net consistently returns the highest IoU scores across all case studies (Table 3). Notably, the U-Net model achieves a mean IoU score of 0.752, outperforming the other models significantly. DeepLabV3 outputs
a mean IoU score of 0.676, and Mask R-CNN achieves 0.512. As the sample size decreases to 256 pixels, the mean IoU scores drop for U-Net (0.665) and DeepLabV3 (0.478), while the Mask R-CNN models output a higher score than before (0.551). Mask R-CNN and DeepLabV3 display relatively close performance in all case studies but also yield the lowest scores observed in this paper. For instance, DeepLabV3 achieves a score of 0.057 in the Missouri area, while Mask R-CNN scores 0.15 in the same region. Reducing the sample size further to 128 pixels results in increased mean IoU scores for all models. U-Net continues to deliver high scores across the cases examined, maintaining a robust mean score of 0.763, consistent with the performance obtained from training on the 512-pixel sample size. Interestingly, both DeepLabV3 and Mask R-CNN show improved performance in this scenario compared to their performance with the 512- and 256-pixel sample sizes. This improvement suggests that they benefit from a large input dataset albeit at a smaller size. Still, a large sample size such as $512 \times 512$ pixels is expected to return better results, as it contains more heterogeneous information and better spatial delineation of the phenomena (Sandric et al., 2024).

The selected case studies – Missouri River, Broughton Creek, Răul Negru, Timiș River, and Prut River – offer valuable insights into segmentation model performance across diverse environmental contexts. This selection ensured a thorough evaluation of the models’ robustness and accuracy in different flood events and geographic settings. These case studies encompass a wide range of geographic locations and hydrological characteristics, including large floodplains and narrow river channels. In the case of the Timiș River, confined by dykes and presenting a narrow channel, less accurate results were achieved. This case underscores the challenges segmentation models face in narrow, confined flooding environments compared to large floodplains, where models perform better due to easier segmentation of homogeneously flooded areas. In contrast, the large flooded areas in other case studies generally achieved better results. For example, the Missouri River, with its extensive floodplain, allowed models to perform more accurately due to the easier segmentation of large, homogeneously flooded areas. This difference underscores the importance of geographic context in evaluating segmentation model performance.

**Conclusions**

The aim of this study was to identify advanced detection models suitable for mapping floods efficiently and accurately using Sentinel-1 GRD images. We selected two machine learning algorithms (Random Forest and Support Vector Machine) along with prevalent deep learning algorithms (U-Net, Mask-R-CNN, and DeepLabv3). These algorithms underwent training using samples from five radar satellite images, four before the flood event and one after the flood event, for deep learning and machine learning models. Additionally, we conducted object-oriented analysis with machine learning algorithms, enabling the interpretation of image elements as objects, a common practice in geography, rather than merely pixels. This approach yielded superior results, surpassing even Mask R-CNN and DeepLabv3 in terms of IoU and closely approaching U-Net’s performance.

We tested algorithms trained specifically on a single area, the Prut River region, and discovered their capacity to accurately identify flooded areas within unknown images, even when sourced from disparate global locations and offering a limited number of perspectives. The results highlight the great potential of a relatively new technique that combines artificial intelligence capabilities with C-band SAR satellite data from the European Space Agency’s Copernicus program.

The satellite image’s spatial resolution and the size and shape of flooded surfaces notably impact the accuracy of the models. In particular, in extensive flooded areas, the results consistently displayed higher IoU scores.

Based on the results we obtained from the multi-temporal analyses conducted using the seven methods described in this paper, we believe that the U-Net model is the most feasible to assess with great accuracy the ground impacts of extreme weather or hydrological events swiftly and inexpensively, such as floods, without requiring on-site human intervention.

**Disclosure statement**

No potential conflict of interest was reported by the author(s).

**Data availability statement**

The data that support the findings of this study are available on request from the corresponding author [IS].

**References**

Antzoulatos, G., Kougoglou, I.-O., Bakratsas, M., Moumtzidou, A., Gialampoukidis, I., Karakostas, A., Lombardo, F., Fiorin, R., Norbiato, D., Ferri, M., Symeonidis, A., Vrochidis, S., & Kompatsiaris, I. (2022). Flood hazard and risk mapping by applying an explainable machine learning framework using satellite imagery and GIS data. *Sustainability*, 14(6), 3251. https://doi.org/10.3390/su14063251
Bayik, C., Abdikan, S., Ozbulak, G., Alasag, T., Aydemir, S., & Balık Sanlı, F. (2018). EXPLOITING MULTI-TEMPORAL SENTINEL-1 SAR DATA for FLOOD EXTEND MAPPING. *International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, XLII-3/W4* (March), 109–113. https://doi.org/10.5194/isprs-archives-XLII-3-W4-109-2018

Bonaﬁlia, D., Tellman, B., Anderson, T., & Issenberg, E. (2020). Sen1Floods11: A georeferenced dataset to train and test deep learning flood algorithms for sentinel-1. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops* (pp. 210–211). IEEE.

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

Cao, H., Zhang, H., Wang, C., & Zhang, B. (2019). Operational flood detection using sentinel-1 SAR data over large areas. *Water*, 11(4), 786. https://doi.org/10.3390/w11040786

Carreño Conde, F., & De Mata Muñoz, M. (2019). Flood monitoring based on the study of sentinel-1 SAR images: The Ebro River case study. *Water*, 11(12), 2454. https://doi.org/10.3390/w11122454

Chakma, P., & Akter, A. (2021). Flood mapping in the coastal region of Bangladesh using sentinel-1 SAR images: A case study of super cyclone amphan. *Journal of the Civil Engineering Forum*, 7(3), 267. https://doi.org/10.22146/jcef.64497

Chen, L. C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2018). DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. *IEEE Transactions on Pattern Analysis & Machine Intelligence*, 40(4), 834–848. https://doi.org/10.1109/TPAMI.2017.2699184

Chen, S., Huang, W., Chen, Y., & Feng, M. (2021). An adaptive thresholding approach toward rapid flood coverage extraction from sentinel-1 SAR imagery. *Remote Sensing*, 13(23), 4899. https://doi.org/10.3390/rs13234899

Colacicco, R., Refice, A., Nutricato, R., Bovenga, F., Caporusso, G., D’Addabbo, A., La Salandra, M., Paolo Lovergine, F., Oscar Nitti, D., & Capolongo, D. (2024). High-resolution flood monitoring based on advanced statistical modeling of sentinel-1 multi-temporal stacks. *Remote Sensing*, 16(2), 294. https://doi.org/10.3390/rs16020294

Cortes, C., Vapnik, V., & Saitta, L. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297. https://doi.org/10.1007/BF00994018

Dong, Z., Liang, Z., Wang, G., Obiri Yeboah Amankwah, S., Feng, D., Wei, X., & Duan, Z. (2023). Mapping inundation extents in Poyang Lake area using sentinel-1 data and transformer-based change detection method. *Journal of Hydrology*, 620(May), 129455. https://doi.org/10.1016/j.jhydrol.2023.129455

Drăguț, L., Blaschke, T., & Dragut, L. (2006). Automated classification of landform elements using object-based image analysis. *Geomorphology*, 81(3–4), 330–344. https://doi.org/10.1016/j.geomorph.2006.04.013

Drăguț, L., Tiede, D., & Levick, S. R. (2010). ESP: A tool to estimate scale parameter for multiresolution image segmentation of remotely sensed data. *International Journal of Geographical Information Science*, 24(6), 859–871. https://doi.org/10.1080/13658810903174803

Drakonakis, G. I., Tsagkatakis, G., Fotiadou, K., & Tsakalides, P. (2022). OmbriaNet—supervised flood mapping via convolutional neural networks using multi-temporal sentinel-1 and sentinel-2 data fusion. *IEEE Journal of Selected Topics in Applied Earth Observations & Remote Sensing*, 15, 2341–2356. https://doi.org/10.1109/JSTARS.2022.3155559

Fraccaro, P., Stoyanov, N., Gaffoor, Z., Elena Cue La Rosa, L., Singh, J., Ishikawa, T., Edwards, B., Jones, A., & Weldermariam, K. (2022). Deploying an artificial intelligence application to detect flood from sentinel 1 data. *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(11), 12489–12495. https://doi.org/10.1609/aaai.v36i11.21517

Gašparović, M., & Klobočar, D. (2021). Mapping floods in lowland forest using sentinel-1 and sentinel-2 data and an object-based approach. *Forests*, 12(5), 553. https://doi.org/10.3390/f12050553

Halder, S., & Bose, S. (2024, January). Sustainable flood hazard mapping with GLOF: A Google earth engine approach. *Natural Hazards Research*, S2666592124000027. https://doi.org/10.1016/j.nhres.2024.01.002

Hao, C., Yunus, A. P., Subramanian, S. S., & Avtar, R. (2021). Basin-wide flood depth and exposure mapping from SAR images and machine learning models. *Journal of Environmental Management*, 297(November), 113367. https://doi.org/10.1016/j.jenvman.2021.113367

Hardy, A., Ettritch, G., Cross, D., Bunting, P., Liyalwihi, F., Sakala, J., Silumesi, A., Singini, D., Smith, M., Willis, T., & Thomas, C. J. (2019). Automatic detection of open and vegetated water bodies using sentinel 1 to map African malaria vector mosquito breeding habitats. *Remote Sensing*, 11(5), 593. https://doi.org/10.3390/rs11050593

Huang, M., & Jin, S. (2020). Rapid flood mapping and evaluation with a supervised Classifier and change detection in Shouguang using sentinel-1 SAR and Sentinel-2 optical data. *Remote Sensing*, 12(13), 2073. https://doi.org/10.3390/rs12132073

Huang, Z., Wu, W., Liu, H., Zhang, W., & Hu, J. (2021). Identifying dynamic changes in water surface using sentinel-1 data based on genetic algorithm and machine learning techniques. *Remote Sensing*, 13(18), 3745. https://doi.org/10.3390/rs13183745

Islam, T., MD, & Meng, Q. (2022). An exploratory study of sentinel-1 SAR for rapid urban flood mapping on Google earth engine. *International Journal of Applied Earth Observation and Geoinformation*, 113(September), 103002. https://doi.org/10.1016/j.jag.2022.103002

Jamali, A., Kumar Roy, S., Hashemi Beni, L., Pradhan, B., Li, J., & Ghamsi, P. (2024). Residual wave vision U-Net for flood mapping using dual polarization sentinel-1 SAR imagery. *International Journal of Applied Earth Observation and Geoinformation*, 127(March), 103662. https://doi.org/10.1016/j.jag.2024.103662

Jiang, X., Liang, S., He, X., Ziegler, A. D., Lin, P., Pan, M., Wang, D., Zou, J., Hao, D., Mao, G., Zeng, Y., Yin, J., Feng, L., Miao, C., Wood, E. F., & Zeng, Z. (2021). Rapid and large-scale mapping of flood inundation via integrating spaceborne synthetic aperture radar imagery with unsupervised deep learning. *Isprs Journal of Photogrammetry & Remote Sensing*, 178(August), 36–50. https://doi.org/10.1016/j.isprsjprs.2021.05.019

Kaiming, H., Gkioxari, G., Dollár, P., & Girshick, R. 2018 Mask R-CNN. *arXiv*. http://arxiv.org/abs/1703.06870

Kim, J., Kim, H., Jeon, H., Jeong, S.-H., Song, J., Vadivel, S. K. P., & Kim, D.-J. (2021). Synergistic use of geospatial data for water body extraction from sentinel-1 images for operational flood monitoring across Southeast Asia using deep neural networks. *Remote Sensing*, 13(23), 4759. https://doi.org/10.3390/rs13234759
Klemas, V. (2015). Remote sensing of floods and flood-prone areas: An overview. *Journal of Coastal Research*, 314(July), 1005–1013. https://doi.org/10.2112/JCOASTRES-D-14-00160.1

Konapala, G., Kumar, S. V., & Ahmad, S. K. (2021). Exploring sentinel-1 and sentinel-2 diversity for flood inundation mapping using deep learning. *Isprs Journal of Photogrammetry & Remote Sensing*, 180(October), 163–173. https://doi.org/10.1016/j.isprsjprs.2021.08.016

Landuyt, L., Verhoest, N. E. C., & Van Coillie, F. M. B. (2020). Flood mapping in vegetated areas using an unsupervised clustering approach on sentinel-1 and -2 imagery. *Remote Sensing*, 12(21), 3611. https://doi.org/10.3390/rs12213611

Lin, L., Di, L., Genong Yu, E., Kang, L., Shrestha, R., Shahinovo Rahman, M., Tang, J., Tang, J., Deng M., Sun Z., & Zhang C. (2016). A review of remote sensing in flood assessment. 2016 Fifth International Conference on Agro-Geoinformatics (Agro-Geoinformatics) (pp. 1–4). IEEE, Tianjin, China. https://doi.org/10.1109/Agro-Geoinformatics.2016.7577655

Liu, T., & Abd-Elrahman, A. (2018). An object-based image analysis method for enhancing classification of land covers using fully convolutional networks and multi-view images of small unmanned aerial system. *Remote Sensing*, 10(3). 3. https://doi.org/10.3390/rs10030457

Markert, K. N., Markert, A. M., Mayer, T., Nauman, C., Haag, A., Poortinga, A., Bhandari, B., Thwal, N. S., Kunlamai, T., Chishie, F., Kwant, M., Phongsapan, K., Clinton, N., Towashiraporn, P., & Saah, D. (2020). Comparing sentinel-1 surface water mapping algorithms and radiometric terrain correction processing in Southeast Asia utilizing google earth engine. *Remote Sensing*, 12(15), 2469. https://doi.org/10.3390/rs12152469

Mayer, T., Poortinga, A., Bhandari, B., Nicolau, A. P., Markert, K., Soe Thwal, N., Markert, A., Haag, A., Kilbride, J., Chishie, F., Wadhwa, A., Clinton, N., & Saah, D. (2021). Deep learning approach for sentinel-1 surface water mapping leveraging google earth engine. *ISPRS Open Journal of Photogrammetry and Remote Sensing*, 2(December), 100005. https://doi.org/10.1016/j.isprsophoto.2021.100005

McCormack, T., Campanyà, J., & Naughton, O. (2022). A methodology for mapping annual flood extent using multi-temporal sentinel-1 imagery. *Remote Sensing of Environment*, 282(December), 113273. https://doi.org/10.1016/j.rse.2022.113273

Moharrami, M., Javanbakht, M., & Attarchi, S. (2021). Automatic flood detection using sentinel-1 images on the Google earth engine. *Environmental Monitoring and Assessment*, 193(5), 248. https://doi.org/10.1007/s10661-021-09037-7

Pantazi, X.-E., Tamouridou, A.-A., Moshou, D., Cherif, I., Ovakoglou, G., Tseni, X., Kalaitzopoulou, S., Mourelatos, S., & Alexandridis, T. K. (2022). Evaluation of machine learning approaches for surface water monitoring using sentinel-1 data. *Journal of Applied Remote Sensing*, 16(4). https://doi.org/10.1117/1.JRS.16.044501

Pedzisai, E., Mutanga, O., Odindi, J., & Bangira, T. (2023). A novel change detection and threshold-based ensemble of scenarios pyramid for flood extent mapping using sentinel-1 data. *Heliyon*, 9(3), e13332. https://doi.org/10.1016/j.heliyon.2023.e13332

Rahman, M., Chen, N., Elbeltagi, A., Monirul Islam, M., Alam, M., Reza Pourghasemi, H., Tao, W., Zhang, J., Shufeng, T., Faiz, H., Baig, M. A., & Dewan, A. (2021). Application of stacking hybrid machine learning algorithms in delineating multi-type flooding in Bangladesh. *Journal of Environmental Management*, 295 (October), 113086. https://doi.org/10.1016/j.jenvman.2021.113086

Reksten, J. H., Salberg, A.-B., & Solberg, R. (2019). FLOOD DETECTION in NORWAY BASED on SENTINEL-1 SAR IMAGERY. *International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, XLII-3/W8(August)*, 349–355. https://doi.org/10.5194/isprs-archives-XLII-3-W8-349-2019

Ren, S., He, K., Girshick, R., & Sun, J. (2017). Faster R-CNN: Towards real-time object detection with region proposal networks. *IEEE Transactions on Pattern Analysis & Machine Intelligence*, 39(6), 1137–1149. https://doi.org/10.1109/TPAMI.2016.2577031

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics)*, 9351, 234–241. https://doi.org/10.1007/978-3-319-24574-4_28

Sandric, I., Chitu, Z., Ilinca, V., & Irimia, R. (2024, June). Using high-resolution UAV imagery and artificial intelligence to detect and map landslide cracks automatically. *Landslides*, 21(10), 2535–2543. https://doi.org/10.1007/s10346-024-02295-9

Şandric, I., Irimia, R., Petropoulos, G. P., Anand, A., Srivastava, P. K., Pleşoianu, A., Faraslis, I., Stateras, D., Kalivas, D., & Ples, A. (2022). Tree’s detection & health’s assessment from ultra-high resolution UAV imagery and deep learning, *Geocarto International*, (25), 37. https://doi.org/10.1080/10106049.2022.2036824

Schumann, G. J.-P., & Moller, D. K. (2015). Microwave remote sensing of flood inundation. *Physics and Chemistry of the Earth, Parts A/B/C*, 83–84, 84–95. https://doi.org/10.1016/j.pce.2015.05.002

Shen, X., Wang, D., Mao, K., Anagnostou, E., & Hong, Y. (2019). Inundation extent mapping by synthetic aperture radar: A review. *Remote Sensing*, 11(7), 879. https://doi.org/10.3390/rs11070879

Soria-Ruiz, J., Fernandez-Ordoñez, Y. M., Ambrosio-Ambrosio, J. P., Escalona-Maurice, M. J., Medina-García, G., Sotelo-Ruiz, E. D., & Ramirez-Guzman, M. E. (2022). Flooded extent and depth analysis using optical and SAR remote sensing with machine learning algorithms. *Atmosphere*, 13(11), 1852. https://doi.org/10.3390/atmos13111852

Tavus, B., Kocaman, S., & Gokceoglu, C. (2022). Flood damage assessment with sentinel-1 and sentinel-2 data after Sardoba dam break with GLCM features and random forest method. *Science of the Total Environment*, 816(April), 151585. https://doi.org/10.1016/j.scitotenv.2021.151585

Tay, C. W. J., Yun, S.-H., Tong Chin, S., Bhardwaj, A., Jung, J., & Hill, E. M. (2020). Rapid flood and damage mapping using synthetic aperture radar in response to typhoon Hagibis, Japan. *Scientific Data*, 7(1), 100. https://doi.org/10.1038/s41597-020-0443-5

Tran, K. H., Menenti, M., & Jia, L. (2022). Surface water mapping and flood monitoring in the Mekong delta using sentinel-1 SAR time series and otsu threshold. *Remote Sensing*, 14(22), 5721. https://doi.org/10.3390/rs14225721

Tsyganskaya, V., Martinis, S., Marzahn, P., & Ludwig, R. (2018). Detection of temporary flooded vegetation using sentinel-1 time series data. *Remote Sensing*, 10(8), 1286. https://doi.org/10.3390/rs10081286
Uddin, M., & Meyer, F. J. (2019). Operational flood mapping using multi-temporal sentinel-1 SAR images: A case study from Bangladesh. Remote Sensing, 11(13), 1581. https://doi.org/10.3390/rs11131581

Ulloa, N. I., Yun, S.-H., Chiang, S.-H., & Furuta, R. (2022). Sentinel-1 spatiotemporal simulation using convolutional LSTM for flood mapping. Remote Sensing, 14(2), 246. https://doi.org/10.3390/rs14020246

Wagner, W., Freeman, V., Cao, S., Matgen, P., Chini, M., Salamon, P., McCormick, N., Martinis, S., Bauer-Marschallinger, B., Navacchi, C., Schramm, M., Reimer, C., & Briese, C. (2020, August). Data processing architectures for monitoring floods using Sentinel-1. ISPRS Annals of the Photogrammetry, Remote Sensing & Spatial Information Sciences, V-3–2020, 641–648. https://doi.org/10.5194/isprs-annals-V-3-2020-641-2020

Wang, J., Wang, S., Wang, F., Zhou, Y., Wang, Z., Ji, J., Xiong, Y., & Zhao, Q. (2022). Fwenet: A deep convolutional neural network for flood water body extraction based on SAR images. International Journal of Digital Earth, 15(1), 345–361. https://doi.org/10.1080/17538947.2021.1995513

Yadav, R., Nascenti, A., & Ban, Y. (2022). Attentive dual stream siamese U-Net for flood detection on multi-temporal sentinel-1 data. IGARSS 2022-2022 IEEE International Geoscience and Remote Sensing Symposium, Kuala Lumpur, Malaysia (pp. 5222–5225). https://doi.org/10.1109/IGARSS46834.2022.9883132

Yang, Q., Shen, X., Anagnostou, E. N., Mo, C., Eggleston, J. R., & Kettner, A. J. (2021). A high-resolution flood inundation archive (2016—the present) from sentinel-1 SAR imagery over CONUS. Bulletin of the American Meteorological Society, 102(5), E1064–79. https://doi.org/10.1175/BAMS-D-19-0319.1

Yommy, A. S., Liu, R., & Shuang Wu, A. (2015). SAR image despeckling using refined lee filter. Proceedings - 2015 7th International Conference on Intelligent Human-Machine Systems and Cybernetics, IHMSC 2015 (Vol. 2., pp. 260–265). https://doi.org/10.1109/IHMSC.2015.236

Zhang, M., Chen, F., Liang, D., Tian, B., & Yang, A. (2020). Use of Sentinel-1 GRD SAR images to delineate flood extent in Pakistan. Sustainability, 12(14), 5784. https://doi.org/10.3390/su12145784

Zhang, X., Weng Chan, N., Pan, B., Ge, X., & Yang, H. (2021). Mapping flood by the object-based method using backscattering coefficient and interference coherence of Sentinel-1 time series. Science of the Total Environment, 794(November), 148388. https://doi.org/10.1016/j.scitotenv.2021.148388

Zhao, J., Li, Y., Matgen, P., Pelich, R., Hostache, R., Wagner, W., & Chini, M. (2022). Urban-aware U-Net for large-scale urban flood mapping using multitemporal sentinel-1 intensity and interferometric coherence. IEEE Transactions on Geoscience & Remote Sensing, 60, 1–21. https://doi.org/10.1109/TGRS.2022.3199036