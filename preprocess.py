from torchvision import datasets, transforms

_IMAGENET_NORMALIZE_STATS = {'mean': [0.485, 0.456, 0.406],
                             'std': [0.229, 0.224, 0.225]}

def scale_crop(input_size, scale_size=None, normalize=_IMAGENET_NORMALIZE_STATS):
    convert_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])
    t_list = [
        transforms.CenterCrop(input_size),
        convert_tensor
    ]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)

def preprocess_imagenet(input_size, normalize=_IMAGENET_NORMALIZE_STATS):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])

def get_transform(transform_name='imagenet', augment=True, input_size=None,
                  scale_size=None, normalize=None):
    normalize = normalize or _IMAGENET_NORMALIZE_STATS
    if transform_name == 'mnist':
        normalize = {'mean': [0.5], 'std': [0.5]}
        input_size = input_size or 28
        scale_size = scale_size or 32
        return scale_crop(input_size=input_size, scale_size=scale_size,
                          normalize=normalize)

    elif 'cifar' in transform_name:
        input_size = input_size or 32
        scale_size = scale_size or 32
        return scale_crop(input_size=input_size, scale_size=scale_size,
                          normalize=normalize)
    
    elif transform_name == 'imagenet' or transform_name == 'imagenet_blurred':
        input_size = input_size or 224
        scale_size = scale_size or int(input_size * 8 / 7)
        if augment:
            return preprocess_imagenet(input_size=input_size,
                                       normalize=normalize)
        else:
            return scale_crop(input_size=input_size, scale_size=scale_size,
                              normalize=normalize)

    elif transform_name == 'tinyimagenet':
        input_size = input_size or 64
        scale_size = scale_size or int(input_size * 8 / 7)
        if augment:
            return preprocess_imagenet(input_size=input_size,
                                       normalize=normalize)
        else:
            return scale_crop(input_size=input_size, scale_size=scale_size,
                              normalize=normalize)