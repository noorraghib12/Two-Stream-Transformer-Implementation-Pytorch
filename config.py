def get_config():
    return dict(
        imgsz= 300,
        n_patches= 36,
        max_seql=50,
        d_model=768,
        nhead=8,
        dim_feedforward=2048,
        num_ts_blocks=4,
        num_epochs=20,
        model='model',
        model_folder='weights',
        model_filename='tmodel_'
    )

class config:
    def __init__(self):
        self.__dict__=get_config()




