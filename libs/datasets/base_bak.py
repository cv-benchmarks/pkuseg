from abc import ABCMeta, abstractmethod
from torch.utils import data
import transforms


class BaseDataset(data.Dataset, metaclass=ABCMeta):
    def __init__(self, list_path, data_root, trans_config):
        super(BaseDataset, self).__init__()
        self.list_path = list_path
        self.data_root = data_root

        for name in trans_config.get('names', []):
            assert name in transforms.__all__
        self.trans_config = trans_config

        self.file_list = self.read_list()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path_pair = self.file_list[idx]
        image, label = self.fetch_pair(path_pair)
        _, image_name = path_pair[0].rsplit('/', 1)

        names = self.trans_config.get('names', [])
        configs = self.trans_config.get('configs', dict())

        for name in names:
            transform = getattr(transforms, name)
            image, label = transform(image, label, **configs.get(name, dict()))

        image, label = transforms.to_tensor(image, label)
        return image, label, image_name

    @abstractmethod
    def read_list(self):
        return None

    @abstractmethod
    def fetch_pair(self, path_pair):
        return None, None
