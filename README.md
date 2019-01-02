# faiss-for-images-duplicate-remove
faiss for images duplicate remove 

## step

### step1

首先利用现有的框架对每张图片提取特征并保存，可以是用caffe，pytorch或者其他框架，比如提取倒数第二层全连接层，假设维度为4096
### step2

保存所有图片的特征存为npy文件
大小为n（张）*4096
### step3

修改search_for_remove_deplicate.py对应的npy文件
### step4

修改search_for_remove_deplicate.py中设定的相似图片张数（比如总共1000张图片，每张图片查找与其相似的99张，则每行list表示与该张图片相似的图片的索引），最终返回的为10*100的array，也就是每100（1张查询图+99张相似图）为一组，共有10000/10=10组
## Usage 
python search_for_remove_deplicate.py
