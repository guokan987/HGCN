# HGCN
Updata 1:

2022/7/18
Please Emails:guokan.cn@gmail.com to contact me;
邮箱地址以更换为：guokan.cn@gmail.com。

Update:

2021/5/11 14:58
The original data, road GPS Coordinate, and code of Generate Data have updated in the Dataset by Baidu Cloud.
原始的数据，具体的路段起始节点GPS坐标(百度坐标参考系)和生成数据和邻接矩阵的代码已经在数据中更新，直接重新访问分享的百度网盘连接，即可下载食用 (.V.)。。。。。。

分割线_________________________________________________________分割线

The code of AAAI2021 paper of HGCN for Traffic Forecasting

This is a document for this code++++++++++++++++++++++++++++++++++++++++++++++++++++++++=>

***First, the structure of the code:

util.py: The reading of data and other functions except of Model

utils.py: all base blocks and functions of neural network models in our paper

model.py: construct model based on utils.py

engine.py: the program of training model in our paper

train.py: compared models which doesn't deal with Region data

train_h.py: HGCN and HGCN_wdf

***How to run these files?

In jupyter ,you should write:

run train.py --model (you can select, such as gwnet) --force True

or 

run train_h.py --model (you can select, such as H_GCN) --force True

If you want to change the dataset from XiAn_city to JiNan_city, I suggest you can directly revise the code in your IDE in train.py or train_h.py

***The dataset problem:

Attention:

Because the DiDi company's data pact, we cannot publish the original or other data, So if you want to get data, you should first apply data in https://outreach.didichuxing.com/app-vue/TTI?id=1003, and eamil me, we get your this paper's data by Baidu Cloud website and key.

由于滴滴公司的协议，我们不能直接公开数据，所以，如果想使用数据，需要先在上面的网站申请，得到许可后，您可以选择使用原始数据(比较乱，定位不准，所以我们重新定位了道路起始点，构建了路网的邻接矩阵)，或者联系我，我直接给您论文中使用的已经清洗好的格式化数据。

Please leave a message in the issue area with English or Chinese. You can also email me!

***The each epoch's save Pytorch model (stat_dict) file in the following website, if you are interested in each epoch's details in our model training phase, please get the authority of data in above website. and we give your key in email.

每一个模型的训练epoch 的参数已经加密上传到百度网盘，如您需要，在得到数据许可后，我们提供密码。
weibsite：链接：https://pan.baidu.com/s/1kYNF4EOYwI2EF2c5xuTxQw 
提取码：_____.

