def get_config():
    return dict(
        imgsz= 300,
        n_patches= 36,
        max_seql=50,
        d_model=768,
        nhead=8,
        dim_feedforward=2048,
        num_ts_blocks=4,
        tokenizer_model='medicalai/ClinicalBERT',
        num_epochs=20,
        device='cpu',
        train_batchsz=10,
        val_batchsz=10,
    )

class config:
    def __init__(self):
        self.__dict__=get_config()




