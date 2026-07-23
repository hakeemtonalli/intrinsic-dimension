from itertools import islice
import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
from torchvision import transforms


def preprocess_text(text):
    return text.replace("\n", " ").replace("  ", " ")


def preprocess_image(image):
    """Preprocess the image for ResNet."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image)


def download_comments(url, num_comments=20):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
    comments_list = []
    for comment in islice(comments, num_comments):
        # print(comment)
        comments_list.append(comment)

    comments_df = pd.DataFrame(comments_list)
    return comments_df
