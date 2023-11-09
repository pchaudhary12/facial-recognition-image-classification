# Facial-Expression-Recognition-Image-Classification

## Business Understanding

--> Facial Expression Recognition is a critical technology that is gaining traction in various industries, including healthcare, entertainment, and security. The goal of this project is to develop an accurate and efficient Facial Expression Recognition model using the PyTorch framework. The model will be trained on a large dataset of facial expressions to recognize different emotions, including happiness, sadness and neutral.

--> The ultimate objective of this project is to provide businesses with a tool that can improve customer experience, increase security, and enhance overall operational efficiency. For instance, the technology can be used in healthcare to detect early signs of depression or anxiety in patients, in entertainment to enhance gaming experiences, and in security to monitor public spaces for suspicious behavior. The Facial Expression Recognition with PyTorch project has the potential to revolutionize various industries by providing an automated, accurate, and reliable tool for recognizing emotions.

## Understanding Dataset

--> The Face Expression Recognition dataset available on Kaggle contains 28,709 images of human faces, labeled with seven different facial expressions, including angry, disgust, fear, happy, sad, surprise, and neutral.  (https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

--> We have taken only three classes in consideration (happy, sad and nuetral) due to limited time for the project, as initial with 7 classes the models were taking too long to train on all 7 classes and were giving low accuracies.

--> The dataset is split into two subsets, a training set of 24,706 images and a test set of 4,003 images. The images are grayscale, 48×48 pixels in size, and the data is stored in CSV format. Each row of the CSV file corresponds to an image and contains the pixel values of the image, the emotion label, and other attributes, such as the image usage and intensity.

--> The dataset is well-balanced, with each emotion class containing approximately the same number of images. It is important to note that the images were extracted from the FER2013 dataset and preprocessed to contain only faces with frontal pose and appropriate brightness, resulting in some loss of information.

--> Additionally, the dataset contains some images with low resolution or artifacts, which may affect the performance of the model.

--> Overall, the Face Expression Recognition dataset provides a diverse and labeled set of images for training and testing facial expression recognition models.