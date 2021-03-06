class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/home/katou/Python/GitHubRepo/Data/VOC2012/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif database == 'sbd':
            return '/path/to/Segmentation/benchmark_RELEASE/' # folder that contains dataset/.
        elif database == 'cityscapes':
            return '/path/to/Segmentation/cityscapes/'         # foler that contains leftImg8bit/
        elif database == 'salt_id':
            return '/home/katou/Python/GitHubRepo/Data/Kaggle Salt Id/train'         # foler that contains leftImg8bit/
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError