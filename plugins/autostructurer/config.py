class Config:
    TEXT_EMBED_MODEL = "all-MiniLM-L6-v2"
    CLIP_MODEL = "openai/clip-vit-base-patch32"

    IVF_NLIST = 4096
    IVF_M = 32
    IVF_NBITS = 8

    TRAIN_MIN_VECTORS = 5000
    BATCH_TEXT = 128
    BATCH_CLIP = 32