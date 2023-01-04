import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# transform 먹인거 보는 코드.
def visualize_aug(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([
        t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))
    ])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12,6))

    for i in range(samples) :
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    
    plt.tight_layout()
    plt.show()


# seed 고정 코드
'''
random_seed =44444
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
'''