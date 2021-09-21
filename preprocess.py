from torchvision import datasets, transforms

def get_transform(transform_name='mnist'):
    if transform_name == 'mnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif transform_name == 'cifar10':
        return transforms.Compose([
            transforms.ToTensor()
        ])
    elif transform_name == 'cifar100':
        return transforms.Compose([
            transforms.ToTensor()
        ])
    elif transform_name == 'imagenet' or transform_name == 'imagenet_blurred':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])