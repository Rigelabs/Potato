import torch
import torch.nn as nn #intorudcing non-linearity
import torch.optim as optim
import torch.utils
from torchvision import datasets,transforms,models
import os



if __name__=='__main__':
    #define data transformations for data augmentation and normalization
    data_transforms = {
        'train':transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    #Define the data directory
    data_dir = 'dataset'
    #create data loaders
    image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ["train","val"]}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],batch_size=16,shuffle=True,num_workers=1) for x in ["train","val"]}
    dataset_sizes = {x:len(image_datasets[x]) for x in ['train','val']}
    print(dataset_sizes)

    class_names =image_datasets['train'].classes
    print(class_names)

    #load the pre-trained Resnet-18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    #Freeze all layers except the final classification layer
    for name,param in model.named_parameters():
        if "fc" in name: #Unfreeze the final classification layer
            param.requires_grad = True
        else:
            param.requires_grad=False
    #Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

    #move the model to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #training loop
    
    num_epochs = 20
    for epoch in range(num_epochs):
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
        
            running_loss =0.0
            running_corrects = 0

            for inputs,labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase =='train'):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs,1)
                    loss = loss_function(outputs,labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds ==labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'Epoch {epoch} Phase: {phase}')    
            print(f'{phase} Loss : {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}')

    print("Training completed")
    
    #savethe model
    torch.save(model.state_dict(),'potato_leaf_diseases.pth')
