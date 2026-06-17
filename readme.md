## Preprocess
```shell
mkdir data
wget http://www.statmt.org/europarl/v7/europarl.tgz
tar zxvf europarl.tgz
python preprocess_text.py
```

## Train
```shell
python main.py 
```
### Notes
+ Please carefully set the $\lambda$ of mutual information part since I have tested the model in different platform, 
i.e., Tensorflow and Pytorch, same $\lambda$ shows different performance.  

## Evaluation
```shell
python performance.py
```
### Notes
+ If you want to compute the sentence similarity, please download the bert model.
