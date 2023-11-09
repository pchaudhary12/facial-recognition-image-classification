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

## Model

### Approach: Transfer Learning

--> Pre-trained Model: A pre-trained model is a neural network that has been trained on a large dataset, usually for a generic task like image classification, object detection, or natural language understanding. These models have learned to extract useful features from data.

--> Fine-Tuning: Instead of training a neural network from scratch, you take a pre-trained model and modify it by adding or changing the final layers, which are specific to the new task you want to solve. This process is called fine-tuning.

--> Training: The modified model is then trained on a smaller dataset for the specific task you're interested in. Since the model has already learned useful features from the large dataset it was initially trained on, it can converge more quickly and effectively on the new task.

--> We are training our model on EfficientNet_B0 (smaller model for faster computation and accuracy)

--> Let's base the model on EfficientNetB0. We'll include the option to swap out average pooling for max pooling and another option to freeze the backbone. As a default we'll assume we want to use average pooling and finetune the backbone feature layers.

```python

# model = Net()
freeze_backbone = False
use_max_pooling = False

model = models.efficientnet_b0(pretrained=True)

if use_max_pooling:
    maxpools = [k.split('.') for k, m in model.named_modules() if type(m).__name__ == 'AdaptiveAvgPool2d']
    for *parent, k in maxpools:
        setattr(model.get_submodule('.'.join(parent)),'avgpool',nn.AdaptiveMaxPool2d(output_size=1))

if freeze_backbone:
    for params in model.parameters():
        params.requires_grad = False

model.classifier = nn.Sequential(
    # nn.BatchNorm1d(num_features=1536,momentum=0.95),
    nn.Linear(in_features=1280, out_features=512),
    nn.ReLU(),
    # nn.Dropout(0.3),
    # nn.BatchNorm1d(num_features=512,momentum=0.95),
    #nn.Linear(in_features=512, out_features=512),
    #nn.ReLU(),
    # nn.Dropout(0.3),
    nn.Linear(in_features=512,out_features=3),
    nn.Softmax()
)

print(model)

summary(model.cuda(),(3, 160, 160))

```

## Training

--> Now we can train the network -- we'll use the Adam optimizer. Note we'll compute the accuracy based on the maximum output value, and track both accuracy and loss over each epoch. We'll also schedule learning rate decay.

```python
lr = 0.0005
lr_ratio = 0.5
patience = 3

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience=patience,factor=lr_ratio)

if torch.cuda.is_available(): # Checking if we can use GPU
    print("USING GPU")
    model = model.cuda()
    criterion = criterion.cuda()
```

#### We have trained the model on total of 50 epochs, its starts overfitting after 80 percent testing accuracy so training more epochs on the same parameters will give the same result

---> For complete training code check: https://github.com/pchaudhary12/facial-recognition-image-classification/tree/main/project_code


## Evaluation

![Alt text](https://github.com/pchaudhary12/facial-recognition-image-classification/blob/main/images/image.png)

--> As seen the model testing accuracy is not going over 80 percent where the training accuracy is going over 90, this is sign of overfitting as testing accuracy is stuck at 80 percent, however it is still considerable testing accuracy.

## Running Prediction after creating the model

--> We are now doing prediction on the unintroducted test image samples after saving the model, to see if the prediction is consistant.

![Alt text](https://github.com/pchaudhary12/facial-recognition-image-classification/blob/main/images/image1.png)

--> We are getting 80.7 Accuracy with this prediction.

--> As seen from the precision and confusion matrix, the model is performing very well with class: Happy, and have 70 percent precision with both Neutral and Sad, as it is understandable that the facial features are similar with both the expression

## Room for Improvement

--> Model accuracy can be improved with more time

--> Try training model with different parameters for efficiency.

--> Preprocessing the image data in way that it  only takes facial Landmarks that explains the expression on the face to imrpove the model.

--> Can try larger pre-trained model if have more computation power.



