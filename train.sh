# 当cluster小于q时的实验没有做


#验证先var再diversity好、还是先diversity再var好

#验证不同的策略结合：是union、intersection好，还是不同先后比较好

# # 先按照var选5q个，再GMCLuster选q个
# #史诗级bug：应该把idx转到aux_dataloader上,影响的实验：KmeansConsistencyQuery、KmeansVarQuery、VarKmeansQuery、ClassVarVarQuery
# python3 train.py -c Config/al-isic-final/CosineEntropy.yml   -s Config/strategy.yml
# python3 train.py -c Config/al-isic-final/MahalanobisEntropy.yml   -s Config/strategy.yml


#var有问题，KmeansVarQuery、ClassVarVarQuery、MahalanobisDistanceVar
python3 train.py -c /home/yeep/project/py/deeplearning/AL-ACDC/Config/al-isic-final/downsamper_cosine/ClassFeatureEntropy.yml   -s Config/strategy.yml
python3 train.py -c /home/yeep/project/py/deeplearning/AL-ACDC/Config/al-isic-final/downsamper_cosine/ClassFeatureLC.yml  -s Config/strategy.yml
python3 train.py -c /home/yeep/project/py/deeplearning/AL-ACDC/Config/al-isic-final/downsamper_cosine/ClassFeatureMC.yml  -s Config/strategy.yml
