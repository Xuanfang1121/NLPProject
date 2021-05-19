# 命名实体识别
本项目是利用hugging face的transformers实现命名实体识别,支持中文bert和roberta-wwm

## requirements
```
tensorflow==2.4.0
transformers==4.2.0
```

## 代码结构
```
.
|__train.py
|__model.py
|__predict.py
|__utils.py
|__app.py
```
说明：
```
1. 训练执行 train.py
2. 预测执行 predict.py
3. http服务请求执行 app.py
```