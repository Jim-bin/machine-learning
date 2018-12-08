# -*- coding: utf-8 -*-

'''
纯Python实现机器学习-感知机模型：适用于线性可分的二分类问题
判断依据为：
对误分类的点必须存在y(i)*(w.x+b)=<0
当对所有样本都有y(i)*(w.x+b)>0时，损失函数等于0，样本完全分开
'''

# 输入数据和标签，输入和标签由元组组成，各个元素构成一个列表
inputData = [((0,0), -1),((1,0), -1), ((0,1), -1), ((2,0), -1), ((1,2), 1), ((2,2), 1), ((2,1), 1), ((2.5,0), 1)]
'''
第一个样本为：
x(1) = (0,0)  inputData[0][0]
y(1) = -1     inputData[0][1]
'''

# l是列表的长度，也就是样本的数目
l = len(inputData)

# 初始化权重、偏差、学习速率
w0 = [0, 0]
b0 = 0
alpha = 0.5

# 迭代函数
def perceptron():
    global b0 # 要在函数中更新函数体之外的变量，python中变量必须设置为全局变量
    global w0
    for i in range(l): # 逐个样本判断是否y(i)*(w.x+b)大于0
        J = inputData[i][1] * (inputData[i][0][0] * w0[0] + inputData[i][0][1] * w0[1] + b0) # 损失函数y(i)*(w.x+b)
        if(J <= 0): # 如果J<0，则表明该样本被误分类了，就在该点处更新w和b，更新策略为梯度下降方法
            w0[0] = w0[0] + alpha * inputData[i][1] * inputData[i][0][0]  # w = w + alpha * y(i)x
            w0[1] = w0[1] + alpha * inputData[i][1] * inputData[i][0][1] 
            b0 = b0 + alpha * inputData[i][1]                             # b = b + alpha *y(i)
            print(w0,b0)
            
            perceptron() # 回调迭代函数，重新判断每个样本，直到每个样本被正确分类，最后返回w和b
    
    return (w0, b0)
    
perceptron()
