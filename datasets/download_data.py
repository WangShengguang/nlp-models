from pathlib import Path

import requests
from tqdm import tqdm


def download(url, file_name=None):
    if file_name is None:
        file_name = Path(url).name
    resp = requests.get(url=url, stream=True)
    # stream=True的作用是仅让响应头被下载，连接保持打开状态，
    content_size = int(resp.headers['Content-Length']) / 1024  # 确定整个安装包的大小
    with open(file_name, "wb") as f:
        print("安装包整个大小是：", content_size, 'k，开始下载...')  # 1024 bytes = 1k ; 1024 k= 1kb
        for data in tqdm(iterable=resp.iter_content(1024), total=content_size, unit='k', desc=file_name):
            # 调用iter_content，一块一块的遍历要下载的内容，搭配stream=True，此时才开始真正的下载
            # iterable：可迭代的进度条 total：总的迭代次数 desc：进度条的前缀
            f.write(data)
        print(file_name + "已经下载完毕！")


if __name__ == '__main__':
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"

    download(url)
