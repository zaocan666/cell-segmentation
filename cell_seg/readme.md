# 细胞分割
## 环境
pytorch 1.4.0
tensorboardx 2.0
tqdm 4.15.0
scikit-image 0.15.0
numpy 1.18.1
cudnn 7.6.5
cudatoolkit 10.1.243

## 网盘地址
https://cloud.tsinghua.edu.cn/d/913cfdbe1d8f47279d91/

## dataset1
### 训练
python run46_40/train.py --root_dir path_to_dataset1
如：
运行：
python run46_40/train.py --root_dir /home/kaixi_whli/segment/supplementary_modify/dataset1

### 预测
从网盘 pretrained model/dataset1 中下载 checkpoints.zip 解压
运行：
python ensemble07/ensemble_submit.py --model_base_path path_to_checkpoints --image_path path_to_test_imgs
如：
python ensemble07/ensemble_submit.py --model_base_path /home/kaixi_whli/segment/data1/run46_40/checkpoints --image_path /home/kaixi_whli/segment/supplementary_modify/dataset1/test

## dataset2
### 在手动标注的数据集上训练
从网盘 dataset 中下载 dataset2.zip 解压
运行：
python run08_ten/train.py --root_dir path_to_dataset2
如：
python run08_ten/train.py --root_dir /home/kaixi_whli/segment/supplementary_modify/dataset2

### 在 DIC-Hela 数据集上训练
从网盘 dataset 中下载 DIC_HELA.zip 解压
运行：
python run12_dic/train.py --root_dir path_to_dic
如：
python run12_dic/train.py --root_dir /home/kaixi_whli/segment/supplementary_modify/DIC_HELA

### 使用手动标注数据训练的模型进行预测
从网盘 pretrained model/dataset2_手标 中下载 checkpoints.zip 解压到 run08_ten 目录下
运行：
python run08_ten/submit.py --image_path path_to_test_imgs
如：
python run08_ten/submit.py --image_path /home/kaixi_whli/segment/supplementary_modify/dataset2/test

### 使用DIC-Hela数据训练的模型进行预测
从网盘 pretrained model/dataset2_dic 中下载 checkpoints.zip 解压到 run12_dic 目录下
运行：
python run12_dic/submit.py --image_path path_to_test_imgs
如：
python run12_dic/submit.py --image_path /home/kaixi_whli/segment/supplementary_modify/dataset2/test

## 文件夹说明
- run46_40：dataset1 训练用代码
- ensemble07：dataset1 模型集成预测所用代码
- run08_ten：dataset2 使用手标数据集训练用代码
- run12_dic：dataset2 使用DIC-Hela数据集训练用代码

## 文件说明
- train.py：训练主逻辑
- eval.py：在验证集上计算IoU；jaccard计算函数
- instance.py：实例化后处理代码
- predict.py：在验证集上计算jaccard分数
- submit.py：在测试集上进行预测
- dice_loss.py：损失函数dice loss计算
- utils/dataset.py：数据集加载以及数据预处理
- utils/cross_eval.py：将训练数据集分成多份，记录到data_list.txt里
- unet：模型代码
- runs：训练过程记录

## 参考代码
https://github.com/milesial/Pytorch-UNet
https://www.aiuai.cn/aifarm1356.html