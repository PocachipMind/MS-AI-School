import albumentations as A
import cv2

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
transform = A.Compose([
        A.ToTensor(),
        A.RandomRotation(180, fill=(0, 0, 0), expand=True),
        A.RandomErasing(scale=(0.025, 0.025), ratio=(1, 1), value=(0, 0, 0)),
        A.RandomPerspective(fill=(0, 0, 0)),
        # transforms.ColorJitter(brightness=.5),
        A.Resize((200, 300)),
        A.RandomCrop((180, 240)),
        A.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)),
        A.RandomHorizontalFlip(p=0.7),
        A.RandomVerticalFlip(p=0.7),
        A.RandomHorizontalFlip(),
        A.Resize((224, 224)),
    ])

image = cv2.imread("/path/to/image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


transformed = transform(image=image)

transformed_image = transformed["image"]

# another_transformed_image = transform(image=another_image)["image"]