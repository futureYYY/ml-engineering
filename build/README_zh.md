# README - 中文翻译

# 构建书籍

重要：这仍然是一个进行中的项目——它大部分功能已经实现，但样式表需要一些工作才能使PDF看起来更好。几周内应该可以完成。

本文档假设你正在从仓库的根目录开始工作。

## 安装要求

1. 安装构建书籍过程中使用的Python包
   ```sh
   pip install -r build/requirements.txt
   ```

2. 下载免费版本的[Prince XML](https://www.princexml.com/download/)。它用于构建此书的PDF版本。

## 构建HTML

```sh
make html
```

## 构建PDF

```sh
make pdf
```

它会先构建HTML目标，然后使用该目标来构建PDF版本。

## 检查链接和锚点

要验证所有本地链接和锚点链接是否有效，请运行：
```sh
make check-links-local
```

要额外检查外部链接，请运行
```sh
make check-links-all
```
但要谨慎使用后者以避免因频繁请求服务器而被封禁。

## 移动md文件/目录并调整相对链接

例如 `slurm` => `orchestration/slurm`
```sh
src=slurm
dst=orchestration/slurm

mkdir -p orchestration
git mv $src $dst
perl -pi -e "s|$src|$dst|" chapters-md.txt
python build/mdbook/mv-links.py $src $dst
git checkout $dst
make check-links-local
```

## 调整图像大小

当包含的图像太大时，将其缩小一点：

```sh
mogrify -format png -resize 1024x1024\> *png
```