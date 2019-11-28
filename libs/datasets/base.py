from abc import ABCMeta, abstractmethod
from torch.utils import data
import transforms


class BaseDataset(data.Dataset, metaclass=ABCMeta):
    def __init__(self, list_path, data_root, trans_types, trans_args):
        super(BaseDataset, self).__init__()
        self.list_path = list_path
        self.data_root = data_root
        self.trans_types = trans_types
        self.trans_args = trans_args

        for func_name in trans_types:
            assert func_name in transforms.__all__

        self.file_list = self.read_list()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path_pair = self.file_list[idx]
        image, label = self.fetch_pair(path_pair)
        _, image_name = path_pair[0].rsplit('/', 1)

        for func_name in self.trans_types:
            transform = getattr(transforms, func_name)
            image, label = transform(image, label, 
                                     **self.trans_args.get(func_name, dict()))

        image, label = transforms.to_tensor(image, label)
        return image, label, image_name

    @abstractmethod
    def read_list(self):
        return None

    @abstractmethod
    def fetch_pair(self, path_pair):
        return None, None
