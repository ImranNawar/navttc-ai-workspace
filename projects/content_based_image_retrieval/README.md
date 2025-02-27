# CBIR (Content Based Image Retrieval)

## Image retrieval methods
- Image retrieval is the process of finding images that match a given query, such as a keyword, a text description, or another image. 
- There are two main methods for image retrieval:\
 `text-based and content-based.` 
- Text-based image retrieval relies on metadata, such as captions, tags, or filenames, to describe the images. 
- Content-based image retrieval uses features, such as color, texture, shape, or deep learning models, to represent the images.

## Content-Based Image Retrieval:
- CBIR is a technique used to retrieve images from a database based on their visual content rather than relying on textual annotations or metadata. 
- CBIR systems analyze the visual features of images, such as color, texture, shape, and spatial arrangement, to find images that are similar to a given query image.
- Content-based image retrieval appears to overcome the disadvantages of the text-based image retrieval methods (searching image databases through the use of Keywords). 
- Text-based image retrieval suffers from many disadvantages:\
           1) manually annotating large databases is not feasible\
           2) subject to human perception,\
           3) these annotations are applicable for only one language.

## Block diagram of CBIR system:
![image](./block_diagram.png)

## Principles and methodologies commonly used in CBIR:
**1) Feature Extraction:** CBIR systems start by extracting relevant features from images.
Common feature extraction techniques include:
- `Color Descriptors:` Represent the distribution of colors in an image.
- `Texture Descriptors:` Capture the texture patterns present in an image, such as local binary patterns (LBP).
- `Shape Descriptors:` Encode the shape or contour information of objects in an image, such as Hu moments or Fourier descriptors.
- `Keypoints descriptors`, SIFT, SURF, ORB, used for extracting distinctive keypoints and descriptors from images, which are robust to changes in scale, rotation, and illumination.

**2) Labeling:**
- Labeling techniques in content-based image retrieval involve assigning category labels or tags to feature vectors extracted from images. This step categorizes each feature vector according to predefined categories or classes.
- It enable classification into predefined categories or classes, facilitating targeted retrieval.
- Example: Assigning labels (0: Cricket_ball, 1: Car, 2: Cricket_bat) to the feature vectors extracted from the dataset.
- **Types of Labeling Techniques**
- `Manual Labeling:` Human annotators assign labels based on image content, accurate but time-consuming.
- `Automated Labeling:` Algorithms or models predict labels based on predefined rules or machine learning techniques.

**3) Indexing:** Once the feature vectors are labeled, they need to be indexed to enable efficient retrieval. Indexing is the process of organizing and storing feature vectors in a way that facilitates fast and accurate retrieval.
Various indexing techniques can be employed, including:
- Inverted Indexing: Maps features to the images that contain them.
- Spatial Indexing: Organizes images based on their spatial properties, such as R-trees or KD-trees.
- Locality-Sensitive Hashing (LSH): Groups similar feature vectors together using hash functions.
- Semantic Indexing: Incorporates domain-specific knowledge or semantic information to index images based on their content.
- Additionally, Faiss, an efficient similarity search library, can be utilized for indexing large collections of feature vectors, providing scalability and high performance.

**4) Similarity Measurement:** CBIR systems use a similarity metric to quantify the similarity between the query image and images in the database. Common similarity measures include:
- Euclidean Distance: Measures the straight-line distance between two feature vectors in the feature space.
- Cosine Similarity: Computes the cosine of the angle between two feature vectors, indicating their similarity in direction.
- Hamming Distance: Suitable for binary feature descriptors, such as ORB or binary SIFT descriptors.

**5) Query Processing:** When a user submits a query image, its feature vector is extracted, and the CBIR system retrieves similar images from the database based on the similarity metric. Query processing involves efficiently searching the index to find the most relevant images to the query.


**6) Evaluation Metrics:** To assess the performance of a CBIR system, various evaluation metrics can be used, including precision, recall, F1-score, mean average precision (mAP), and mean reciprocal rank (MRR).

