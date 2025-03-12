## Pneumonia-Classifier

* This pneumonia classifier application takes a chest X-ray image as input and classifies it as either pneumonia or normal.

* The frontend of the application is created with the Python open-source library **Streamlit**.

* For classification, we use **Resnet18**, a deep learning model known for image recognition tasks, which has an accuracy of 80% on our test dataset.

* We selected this model after experimenting with conventional machine learning model, custom cnn architecture, vision transformer pretrained model, and various deep learning pretrained models.

* We achieved the highest accuracy of 86% with the Vision Transformer (ViT), but we selected Resnet18 to balance performance and complexity, Resnet18 is generally faster and more lightweight compared to ViT.

* The application is deployed to Hugging Face Spaces and is now live. You can access it here: [Pneumonia Classifier](https://huggingface.co/spaces/imran-nawar/pneumonia-classifier)


For the complete implementation, documentation, and code samples, please visit the dedicated repository: [Pneumonia-Classifier](https://github.com/ImranNawar/pneumonia-classifier)