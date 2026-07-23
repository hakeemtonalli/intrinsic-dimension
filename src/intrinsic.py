import numpy as np
from tqdm import tqdm
from skdim.id import MLE
import torch
from src.data import preprocess_text


def get_mle_single(model, tokenizer, text, solver):
    inputs = tokenizer(
        preprocess_text(text), truncation=True, max_length=512, return_tensors="pt"
    )
    with torch.no_grad():
        outp = model(**inputs)

    return solver.fit_transform(outp[0][0].numpy()[1:-1])


def get_mle(model, tokenizer, df, key="text", is_list=False):
    dims = []
    MLE_solver = MLE()
    for s in tqdm(df[key]):
        if is_list:
            text = s[0]
        else:
            text = s
        dims.append(get_mle_single(model, tokenizer, text, MLE_solver))

    return np.array(dims).reshape(-1, 1)


def get_vision_mle(embeddings):
    """Compute the MLE for all embeddings."""
    # dims = []
    print("Computing MLE for embeddings...")
    print(embeddings.shape)
    solver = MLE()  # Set n_neighbors to a valid value
    dims = solver.fit_transform(embeddings)
    # for embedding in tqdm(embeddings):
    #     dims.append(get_mle_single(embedding, MLE_solver))
    return np.array(dims).reshape(-1, 1)
