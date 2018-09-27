class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return 'C:\\Users\\SCatheline\\Documents\\GitHub repo\\PASCAL VOC Dataset\\VOCdevkit\\VOC2012\\'  # folder that contains VOCdevkit/.
        elif database == 'sbd':
            return 'C:\\Users\\SCatheline\\Documents\\GitHub repo\\PASCAL VOC Dataset\\benchmark_RELEASE\\' # folder that contains dataset/.
        elif database == 'cityscapes':
            return '/path/to/Segmentation/cityscapes/'         # foler that contains leftImg8bit/
        elif database == 'salt_id':
            return 'C:\\Users\\SCatheline\\Documents\\GitHub repo\\FirstTest\\Kaggle_Challenge_LIVE-master\\data\\train\\'        
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
