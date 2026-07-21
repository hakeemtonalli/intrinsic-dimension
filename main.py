import os
from pathlib import Path
from itertools import islice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# plot sample_mle and cnn_mle in boxplots

import torch
from transformers import RobertaTokenizer, RobertaModel
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
from skdim.id import MLE


def preprocess_text(text):
    return text.replace("\n", " ").replace("  ", " ")


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


def download_comments(url, num_comments=20):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
    comments_list = []
    for comment in islice(comments, num_comments):
        # print(comment)
        comments_list.append(comment)

    comments_df = pd.DataFrame(comments_list)
    return comments_df


def plot_mle(gold_mle, gen_mle):
    # get non-null entries in both
    gold_mle = gold_mle[~np.isnan(gold_mle)]
    gen_mle = gen_mle[~np.isnan(gen_mle)]

    # plot boxplots
    plt.boxplot([gold_mle, gen_mle])
    plt.xlabel("Dataset")
    plt.ylabel("Intrinsic Dimension")


def main():
    ### Loading the model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")

    out_dir = Path("./outputs/")
    out_dir.mkdir(exist_ok=True)
    sample_video = "https://www.youtube.com/watch?v=ScMzIvxBSi4"
    broodwar_video = "https://www.youtube.com/watch?v=q-7nBJr9pG8&t=2070s"
    yann_video = "https://www.youtube.com/watch?v=72Xj8k5WQX4"
    threeblue_video = "https://www.youtube.com/watch?v=aircAruvnKk&vl=en"

    links_dict = {
        "sample_video": sample_video,
        "broodwar_video": broodwar_video,
        "yann_video": yann_video,
        "threeblue_video": threeblue_video,
    }

    print("Downloading comments from videos...")
    sample_df = download_comments(sample_video)
    broodwar_df = download_comments(broodwar_video)
    threeblue_df = download_comments(threeblue_video)
    yann_df = download_comments(yann_video)

    yann_mle = get_mle(model, tokenizer, yann_df, key="text")
    threeblue_mle = get_mle(model, tokenizer, threeblue_df, key="text")
    sample_mle = get_mle(model, tokenizer, sample_df, key="text")
    broodwar_mle = get_mle(model, tokenizer, broodwar_df, key="text")

    intrinsic_dim_dict = {}
    comments_root = os.path.join(out_dir, "videos_comments/")

    for video_name in links_dict.keys():
        print(f"calculating intrinsic dimension for {video_name}")
        comments_dir = os.path.join(comments_root, video_name)
        if not os.path.exists(comments_dir):
            os.makedirs(comments_dir)
        # comments path as csv
        comments_path = os.path.join(comments_dir, "comments.csv")

        # check if comments exist
        if not os.path.exists(comments_path):
            # create dataframe if link doesnt exist
            comments_df = download_comments(links_dict[video_name])
            comments_df.to_csv(comments_path, index=False)
        else:
            # read comments from csv
            comments_df = pd.read_csv(comments_path)

        # get mle
        comments_mle = get_mle(model, tokenizer, comments_df, key="text")
        intrinsic_dim_dict[video_name] = comments_mle

    # reshape vectors to 1D
    intrinsic_dim_dict = {k: v.reshape(-1) for k, v in intrinsic_dim_dict.items()}
    intrinsic_dim_df = pd.DataFrame(intrinsic_dim_dict)

    # boxplot from intrinsic_dim_df
    figs_dir = os.path.join(out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    sns.boxplot(data=intrinsic_dim_df)
    plt.savefig(os.path.join(figs_dir, "intrinsic_dim_boxplot.png"))
    plt.close()

    plot_mle(sample_mle, broodwar_mle)
    plt.title("Sample vs Broodwar")
    plt.savefig(os.path.join(figs_dir, "sample_vs_broodwar.png"))
    plt.close()

    # sample and yann
    plot_mle(sample_mle, yann_mle)
    plt.title("Sample vs Yann")
    plt.savefig(os.path.join(figs_dir, "sample_vs_yann.png"))
    plt.close()

    # sample and three
    plot_mle(sample_mle, threeblue_mle)
    plt.title("Sample vs Three Blue One Brown")
    plt.savefig(os.path.join(figs_dir, "sample_vs_threeblue.png"))
    plt.close()


if __name__ == "__main__":
    main()
