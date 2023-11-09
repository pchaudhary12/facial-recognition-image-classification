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

## Data Loading

--> Here we set up the data loader objects using images stored in folders on the local disk. Note we also set the batch size and apply a series of transforms for data augmentation.

--> The following class is a customized torch dataset class that allows us to specify the transform process (so we can have one transform for training and another for testing).

```python
class MyLazyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)
```

```python
batch_size = 32

xmin_list = [0.2, 0.05, 0.2]
xmax_list = [0.45, 0.25, 0.45]

def get_data():
    data_dir_train = 'C:\\Users\\pruth\\Desktop\\facial-recognition-image-classification\\data\\train'
    data_dir_test = 'C:\\Users\\pruth\\Desktop\\facial-recognition-image-classification\\data\\validation'
   
    transform_train = transforms.Compose([
        #transforms.CenterCrop(100),
        transforms.Resize((160,160)),
        transforms.RandomInvert(0.2),
        transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
        #transforms.ElasticTransform(alpha=20.0,sigma=10.0),
        #transforms.RandomPerspective(),
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    transform_test = transforms.Compose([
        transforms.Resize((160,160)),
        transforms.ToTensor(),
        ])

    data_set_train_raw = datasets.ImageFolder(data_dir_train)
    data_set_test_raw = datasets.ImageFolder(data_dir_test)

    N_data_train = len(data_set_train_raw)
    N_data_test = len(data_set_test_raw)
    
    data_set_train = MyLazyDataset(data_set_train_raw,transform_train)
    data_set_test = MyLazyDataset(data_set_test_raw,transform_test)

    train_idx = list(range(N_data_train))
    np.random.shuffle(train_idx)
    
    test_idx = list(range(N_data_test))
    np.random.shuffle(test_idx)

    data_set_train_sub = torch.utils.data.Subset(data_set_train, indices=train_idx)
    data_set_test_sub = torch.utils.data.Subset(data_set_test, indices=test_idx)
    
    # data_set_train_sub, data_set_test_sub = torch.utils.data.random_split(dataset=data_set, lengths=[N_train, N_test], generator=torch.Generator().manual_seed(42))

    train = DataLoader(data_set_train_sub, batch_size=batch_size, shuffle=True)
    test = DataLoader(data_set_test_sub, batch_size=batch_size, shuffle=False)
 
    
    return train, test

```



