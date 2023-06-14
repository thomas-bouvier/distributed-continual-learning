from torchvision import transforms


_IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}


def get_transform(
    transform_name="imagenet",
    training=True,
    input_size=None,
    normalize=_IMAGENET_STATS,
):
    if transform_name == "mnist":
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=0.5, std=[0.5])]
        )

    elif "cifar" in transform_name:
        if training:
            return transforms.Compose([
                transforms.RandomCrop(input_size or 32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(fillcolor=(128, 128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(**normalize)
            ])
        else:
            return transforms.Compose([
                transforms.RandomCrop(scale_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**normalize)
            ])

    elif (
        transform_name == "imagenet100"
        or transform_name == "imagenet100small"
        or transform_name == "imagenet"
        or transform_name == "imagenet_blurred"
    ):
        if training:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(input_size or 224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize)
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(input_size or 224),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize)
                ]
            )

    elif transform_name == "tinyimagenet":
        if training:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(input_size or 64),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize)
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(input_size or 224 * 8 / 7),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize)
                ]
            )
