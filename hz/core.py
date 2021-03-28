class CoreObject(object):
    @staticmethod
    def create(cls_name, **kwargs):
        if cls_name == 'mongo':
            cls_name = 'hz.db.mongo.MongoDatabase'

        *module_file, cls = cls_name.split('.')
        module_file = '.'.join(module_file)

        _module = __import__(module_file, fromlist=[cls])

        cls = getattr(_module, cls)

        return cls(**kwargs)
