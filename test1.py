#该代码是一个小型的DNN网络代码的实现，并且实现了多种激活函数，并且实现了图形化显示，特别适合直观地理解神经网络的拟合过程
#代码主要测试了sigmoid和relu函数，另外我们还测试了sin和正态分布，这些函数都能很好地拟合函数，但是一定要对初始化权重做一定的处理，否则训练会很难
#原作者：易瑜    邮箱:296721135@qq.com    如果有错误，欢迎指正,如转载，请注明作者和出处
#本代码在python3上执行测试，只依赖两个python库 numpy 和 matplotlib,通过  pip install numpy  和  pip install matplotlib即可安装，非常简单
# matplotlib 绘图的一个连接：http://www.cnblogs.com/webRobot/p/6747386.html
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random
import os
from PIL import Image
import time
class Activation:                   ##子类必须实现下面的函数
    def __init__(self,wRange = 1,hRange = 1,bRandom = True):
        self.setWeightInitParameter(wRange,hRange,bRandom)

    def setWeightInitParameter(self,wRange,hRange,bRandom):
        self.wRange = wRange
        self.hRange = hRange
        self.bRandom = bRandom
    # 初始化权重  wx + b =  w(x + b/w) = w(x + h)  -> h = b/w ,w决定了函数的x方向的缩放，h决定了缩放后x方向的平移
    #初始化权重并不是一个随机初始化的过程，我们测试中发现，在对s型函数拟合的过程中，务必把函数进行合适的缩放，然后初始化偏移，让其均匀地分布在整个整个输入空间
    #但对relu类型的函数，w可以设置为+1，-1即可，只要改变初始偏移即可完成相应的拟合
    def initWeight(self,cell):
        for i in range(len(cell.w)):
            cell.w[i] = self.wRange*(random.choice([1.,-1.]) if self.bRandom else 1)        #random.uniform(-1,1)
        cell.b = (self.hRange*self.wRange)*(random.uniform(-1,1) if self.bRandom else 1)
        if(cell.specialCellType):
            if (self.hRange.__class__ == list or self.hRange.__class__ == np.ndarray)  and not self.bRandom:
                for i in range(len(cell.h)):
                    cell.h[i] = self.hRange[i]
            else:
                for i in range(len(cell.h)):
                    cell.h[i] = (self.hRange) *(random.uniform(-1,1) if self.bRandom else 1)

    def activation_fun(self,x):     #激活函数
        raise NotImplemented("")

    def activation_deri_fun(self,cell):     #偏导
        raise NotImplemented("")

    # 权重差值,求出来的偏导为
    # △loss/△w = deri, （1）
    # 如果令 △w = -speed*deri  （2）
    # 令2代入1可以导出
    #   △loss = deri*△w  = - speed*deri*deri, loss是往恒往小的方向进行的
    #但是这个更新策略并不是唯一的策略，只要令△loss实际是往减小方向的策略理论上都是可以的，比如我们,在deri不为零的前提下
    #令  △w = -speed/deri  （3）
    #代入1,可得 △loss = -speed,  即每更新一步，△loss是以固定速度减小的
    #但是在(3)式的策略其实也可能有一些其他的问题，比如我们的偏导deri只是在当前w的一个很小的邻域内才成立，所以一定要限制△w 的范围，
    #此处是只抛砖引玉，梯度下降的策略很有多种，可以参数一下下面文章：
    #http://www.360doc.com/content/16/1121/12/22755525_608221032.shtml
    def updateDeltaWeight(self,deri,speed,cell,loss,coefficient):
        return -speed * deri


###############################################################X2,梯度很容易爆炸，但可以通过修改更新权重的策略让其拟合一些函数
class ActivationXX(Activation):
        def activation_fun(self, x):  # 激活函数
            if(abs(x) > 1):         #限制x的范围
                x = 1
            return x*x
        def activation_deri_fun(self, cell):  # 偏导
            if(abs(cell.sum) > 1):
                return 0
            return 2*cell.sum

        # def updateDeltaWeight(self,deri,speed,cell,loss,coefficient):            ##权重差值,这种策略貌似会更快一点
        #     sigmoidDri = 2 *abs(cell.sum)
        #     if((sigmoidDri) < 1):      #梯度太小，不处理
        #         return 0.0
        #     if(coefficient > 10):
        #         return 0.0
        #     coefficient = abs(coefficient)
        #     coefficient = max(coefficient,0.1)
        #     maxDelta = (0.3/coefficient)*sigmoidDri                          #一次的x变化不能太大
        #
        #     if abs(deri) > 0.000001:
        #         delta = (speed/deri) * loss
        #     else:
        #         return 0.0
        #     if abs(delta) > maxDelta:
        #         delta = maxDelta if delta > 0 else -maxDelta
        #     return -delta

############################################################### V型函数
class ActivationAbsolute(Activation):
        def activation_fun(self, x):  # 激活函数
            return abs(x)
        def activation_deri_fun(self, cell):  # 偏导
            return 1.0 if cell.sum < 0.0 else 1.0



############################################################### Sinc型函数
class ActivationSinc(Activation):
        def activation_fun(self, x):  # 激活函数
            return 1.0 if x == 0.0 else math.sin(x)/x
        def activation_deri_fun(self, cell):  # 偏导
            x = cell.sum
            return 1.0 if x == 0.0 else math.cos(x)/x - math.sin(x)/(x*x)

class ActivationTanh(Activation):
    def activation_fun(self, x):  # 激活函数
        return math.tanh(x)

    def activation_deri_fun(self, cell):  # 偏导
        return 1 - cell.out*cell.out

class ActivationRelu(Activation):
    def __init__(self,wRange = 1,hRange = 1,bRandom = True,coe = 0):
        super().__init__(wRange=wRange,hRange=hRange,bRandom=bRandom)
        self.radiusCoefficient = coe          #
    def activation_fun(self, x):  # 激活函数
        return max(0.0,x)
    def activation_deri_fun(self, cell):  # 偏导
        return 0.0  if cell.sum <= 0. else 1.0
    def updateDeltaWeight(self,deri,speed,cell,loss,coefficient):
        return -speed * deri/(1 + self.radiusCoefficient*cell.out)

    def setRadiusCoefficient(self,coefficient):
        self.radiusCoefficient = coefficient

class ActivationMyRelu(Activation):    #____/~~~~~~~`,往右移了一下
    def activation_fun(self, x):  # 激活函数
        return max(0.0,x - 0.5)
    def activation_deri_fun(self, cell):  # 偏导
        return 0.0  if cell.sum <= 0. else 1.0

class ActivationLeakyRelu(Activation):
    def activation_fun(self, x):  # 激活函数
        return x if x >= 0.0 else 0.01*x

    def activation_deri_fun(self, cell):  # 偏导
        return 0.01  if cell.sum <= 0  else 1.0

class ActivationStep(Activation):         #___|~~~~~~   ,0  -  1
    def activation_fun(self, x):  # 激活函数
        return 1.0 if x >= 0 else 0

    def activation_deri_fun(self, cell):  # 偏导
        return 0


class ActivationSignum(Activation):         #___|~~~~~~   ,-1  -  1
    def activation_fun(self, x):  # 激活函数
        return 1.0 if x >= 0 else -1.0

    def activation_deri_fun(self, cell):  # 偏导
        return 0.0

class ActivationSoftPlus(Activation):           #ln(1 + e^x)
    def activation_fun(self, x):  # 激活函数
        return math.log(1 + math.exp(x))

    def activation_deri_fun(self, cell):  # 偏导
        return 1/(1 + math.exp(-cell.sum))

class ActivationLecunTanh(Activation):  # LeCun Tanh
    def activation_fun(self, x):  # 激活函数
        return 1.7519*math.tanh(2*x/3)#
    def activation_deri_fun(self, cell):  # 偏导
        return 1.7519*2*(1 - cell.out*cell/(1.7519*1.7519))/3

class ActivationHardTanh(Activation):  #   ____/~~~~~~~~~  ,
    def activation_fun(self, x):  # 激活函数
        return 1 if x > 1.0 else (-1 if x < -1.0 else x)
    def activation_deri_fun(self, cell):  # 偏导
        return  1 if abs(x) < 1.0 else 0

class ActivationArcTan(Activation):  # ArcTan
    def activation_fun(self, x):  # 激活函数
        return math.atan(x)#
    def activation_deri_fun(self, cell):  # 偏导
        return 1 / (cell.sum*cell.sum + 1)

class ActivationSwish(Activation):  # Swish
    def activation_fun(self, x):  # 激活函数
        return x / (1 + math.exp(-x))
    def activation_deri_fun(self, cell):  # 偏导
        return cell.out + (1 - cell.out)/(1 + math.exp(-cell.sum))

class ActivationSoftsign(Activation):  # x/(1 + |x|)
    def activation_fun(self, x):  # 激活函数
        return x/(1 + abs(x))#

    def activation_deri_fun(self, cell):  # 偏导
        return 1 / ((1 + abs(cell.sum))*(1 + abs(cell.sum)))  #


###############################################################sigmoid
class ActivationSigmoid(Activation):
    def activation_fun(self,x):          #激活函数
        try:
            return 1/(1 + math.exp(-x))
        except OverflowError:
            if x < 0.0:
                return 0
            else:
                return 1;

    def activation_deri_fun(self, cell):#偏导
        return cell.out*(1 - cell.out)

    # def updateDeltaWeight(self,deri,speed,cell,loss,coefficient):            ##权重差值,这种策略貌似会更快一点
    #     sigmoidDri = abs(cell.out * (1 - cell.out))
    #     if((sigmoidDri) < 0.1):      #梯度太小，不处理
    #         return 0.0
    #     coefficient = abs(coefficient)
    #     coefficient = max(coefficient,0.1)
    #     maxDelta = (0.3/coefficient)*sigmoidDri                          #一次的x变化不能太大
    #
    #     if abs(deri) > 0.000001:
    #         delta = (speed/deri) * loss
    #     else:
    #         return 0.0
    #     if abs(delta) > maxDelta:
    #         delta = maxDelta if delta > 0 else -maxDelta
    #     return -delta



###############################################################正态分布
class ActivationNormal(Activation):

    def activation_fun(self,x):          #激活函数
        return math.exp(-x*x)

    def activation_deri_fun(self, cell):#偏导
        return -cell.out*2*cell.sum


 ###############################################################tanh(x/2)函数
class ActivationTanh(Activation):
        def activation_fun(self, x):  # 激活函数
            return (1 - math.exp(-x))/(1 + math.exp(-x))

        def activation_deri_fun(self, cell):  # 偏导
            return 0.5*( 1 - cell.out*cell.out)

###############################################################loglog函数
class ActivationLogLog(Activation):

    def activation_fun(self, x):  # 激活函数
        return 1 - math.exp(-math.exp(x))

    def activation_deri_fun(self, cell):  # 偏导
        return math.exp(cell.sum)*cell.out
###############################################################cos函数
class ActivationCos(Activation):
    def activation_fun(self, x):  # 激活函数
        return math.cos(x)

    def activation_deri_fun(self, cell):  # 偏导
        return math.sin(cell.sum)
###############################################################sin函数
class ActivationSin(Activation):
    def initWeight(self, cell):
        for i in range(len(cell.w)):
            cell.w[i] = self.wRange * random.choice([1., -1.])*random.uniform(0.01, 1)
        cell.b = (self.bRange * self.wRange) * random.uniform(-1, 1)

    def activation_fun(self, x):  # 激活函数
            return math.sin(x)

    def activation_deri_fun(self, cell):  # 偏导
        return math.cos(cell.sum)




###############################################################线性函数
class ActivationLiner(Activation):
    def activation_fun(self,x):          #激活函数
        return x

    def activation_deri_fun(self, cell):#偏导
        return 1
    # def updateDeltaWeight(self,deri,speed,cell,loss,coefficient):
    #     return 0.       #暂时先强制为0，测试


########################Cell有两种，一种是以 ∑wi*xi + b 作为输出  ,特殊的是以∑(abs(wi*(xi + hi))/N作为输出
class Cell:
    def __init__(self,activation,specialCellType):
        self._activation = activation
        self.inputCells = None
        self.sum = 0.0
        self.out = 0.0
        self.error = 0.0
        self.specialCellType = specialCellType
        self.extraInput = None
        self.maxOutMultiWeight = 0          #
        self.totalOutMultiWeight = 0
        self.statisticEnable = False
        self.MinusWeightEnable = False
    def setInputCells(self,inputCells):
        self.inputCells = inputCells
        if inputCells.__class__ == list:
            length = len(inputCells)
        else:
            length = inputCells.size

        self.w = [0 for i in range(length)]
        self.delta_w = [0 for i in range(length)]
        if(self.specialCellType):
            self.h = [0 for i in range(length)]
            self.delta_h = [0 for i in range(length)]
        self.b = 0.0
        self.delta_b = 0.0
        if(self._activation):
            self._activation.initWeight(self)

    def setExtraInput(self,extraInput):
        self.extraInput = extraInput
        length = len(extraInput)
        self.w.extend( [0 for i in range(length)])
        self.delta_w.extend([0 for i in range(length)])
        if(self.specialCellType):
            self.h.extend([0 for i in range(length)])
            self.delta_h.extend( [0 for i in range(length)])
        if (self._activation):
            self._activation.initWeight(self)
    def caculateOut(self):                  #计算输出
        sum = 0.0
        i = 0

        lastLayerIsFcLayer = self.inputCells.__class__ == list
        if lastLayerIsFcLayer:
            values = [cell.out for cell  in self.inputCells]
        else:
            (height, width) = self.inputCells.shape
            values = []
            for h in range(height):
                for w in range(width):
                    values.append(self.inputCells[h][w])
        if self.extraInput:
            values.extend(self.extraInput)

        # print("...........extraInput:",self.extraInput)
        # print("w:",self.w,"  b",self.b)


        for lastLayerOut in values:
            outMultiWeight = 0
            if self.specialCellType:
                outMultiWeight = abs(self.w[i] * (lastLayerOut + self.h[i]))
            else:
                outMultiWeight = self.w[i] * lastLayerOut
            if lastLayerIsFcLayer and i < len(self.inputCells) and self.inputCells[i].statisticEnable:          #用来做统计
                self.inputCells[i].totalOutMultiWeight += abs(outMultiWeight)
                self.inputCells[i].maxOutMultiWeight = max(abs(outMultiWeight),self.inputCells[i].maxOutMultiWeight)
            sum += outMultiWeight
            i += 1

        if not self.specialCellType:
            sum += self.b
        else:
            pass#sum = sum/len(values)             #对其求平均
        self.sum = sum
        self.out = self._activation.activation_fun(sum)


    def _clipValue(self,value,clip_min,clip_max):
        if clip_max == clip_min:
            return value
        elif value < clip_min:
            return clip_min
        elif value > clip_max:
            return clip_max
        else:
            return value

    def updateWeight(self,speed,loss,clip_min = 0.,clip_max = 0.):
        if self.inputCells != None:

            i = 0
            outDeri = self.error*self._activation.activation_deri_fun(self)

            if self.inputCells.__class__ == list:
                values = [cell.out for cell in self.inputCells]
            else:
                (height, width) = self.inputCells.shape
                values = []
                for h in range(height):
                    for w in range(width):
                        values.append(self.inputCells[h][w])

            if self.extraInput:
                values.extend(self.extraInput)

            for lastLayerOut in values:
                if self.specialCellType:
                    currSum = self.w[i]*(lastLayerOut + self.h[i])
                    deri = (lastLayerOut + self.h[i])*outDeri#/len(values)
                    if currSum < 0.:
                        deri = -deri
                else:
                    deri = lastLayerOut * outDeri
                self.delta_w[i] = self._activation.updateDeltaWeight(deri,speed,self,loss,lastLayerOut)
                if not self.MinusWeightEnable or self.w[i]*self.delta_w[i] < 0.:
                    self.w[i] += self.delta_w[i]
                self.w[i] = self._clipValue(self.w[i],clip_min,clip_max)
                if self.specialCellType:
                    if abs(self.w[i]) > 0.001:
                        hDeri = outDeri/self.w[i]
                    else:
                        hDeri = outDeri/(0.001 if self.w[i] > 0 else -0.001)    #self.w[i]*outDeri,如果直接按公式用这个，效果非常差，具体原因待查
                    #hDeri /= len(values)
                    if currSum < 0.:        #绝对值，特殊处理一下
                        hDeri = -hDeri;
                    self.delta_h[i] = self._activation.updateDeltaWeight(hDeri,speed,self,loss,1)
                    self.h[i] += self.delta_h[i]
                    self.h[i] = self._clipValue(self.h[i], clip_min, clip_max)
                i += 1
            if not self.specialCellType:
                deri = outDeri
                self.delta_b = self._activation.updateDeltaWeight(deri,speed,self,loss,1)
                self.b += self.delta_b
                # self.b = self._clipValue(self.b, clip_min, clip_max)


class Layer:
    def __init__(self,trainable,activation = None):
        self.trainable = trainable
        self.activation = activation
        self.clip_min = 0.0
        self.clip_max = 0.0

    def setLastLayer(self, lastLayer):
        raise NotImplementedError

    def caculateOut(self):
        raise NotImplementedError

    def setInputAndForward(self, x):  # 仅第一层调用
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def updateWeight(self, speed, loss):
        raise NotImplementedError

    def setTrainable(self,trainable):
        if hasattr(self,'trainable'):
            self.trainable = trainable

    def clip_by_value(self,min_val,max_val):
        self.clip_min = min_val
        self.clip_max = max_val

class FcLayer(Layer):
    def __init__(self,lastLayer = None,cellNum = 1,activation = None,specialCellType = False,trainable = True):
        super().__init__(trainable = trainable,activation = activation)
        self._lastLayer = lastLayer
        self._cellNum = cellNum
        self._specialCellType = specialCellType
        self.cells = [Cell(activation,specialCellType) for i in range(cellNum)]
        self._nextLayer = None
        if lastLayer:
            self.setLastLayer(lastLayer)


    def setLastLayer(self, lastLayer):
        if lastLayer != None and not lastLayer._nextLayer:
            lastLayer._nextLayer = self
            self._lastLayer = lastLayer
            for cell in self.cells:
                cell.setInputCells(lastLayer.cells if hasattr(lastLayer,'cells') else lastLayer.outImage)

    def caculateOut(self):
        for cell in self.cells:
            cell.caculateOut()

    def _forward(self):                      #第一个层调用
        nextLayer = self._nextLayer
        while nextLayer:
            nextLayer.caculateOut()
            nextLayer = nextLayer._nextLayer

    def setInputAndForward(self,x):           #仅第一层调用
        for i in range(len(self.cells)):
            self.cells[i].out = x[i]
        self._forward()

    def fc_backward(self,currLayer,lastLayer):
        for lastLayerCell in lastLayer.cells:
            lastLayerCell.error = 0.0

        for currLayercell in currLayer.cells:
            if hasattr(currLayercell,"error_deri"):             #主要考虑LossEntropy
                deri = currLayercell.error_deri
            else:
                deri = currLayercell._activation.activation_deri_fun(currLayercell) * currLayercell.error

            for j in range(len(lastLayer.cells)):
                lastLayerCell = lastLayer.cells[j]
                lastLayerCell.error += currLayercell.w[j] * deri

    def image_backward(self,currLayer,lastLayer):
        if hasattr(lastLayer,'error'):
            (height,width) = lastLayer.error.shape
            for i in range(height):
                for j in range(width):
                    lastLayer.error[i][j] = 0.0

            for currLayercell in currLayer.cells:
                deri = currLayercell._activation.activation_deri_fun(currLayercell) * currLayercell.error
                for j in range(lastLayer.error.size):
                   # print("deri:",deri,'..w:',currLayercell.w[j],"..b:", currLayercell.b,'error:',currLayercell.error,'sum:',currLayercell.sum)
                    lastLayer.error[j//width][j%width] += currLayercell.w[j] * deri
            # print("error:",lastLayer.error)


    def backward(self):
        if hasattr(self._lastLayer, 'cells'):
            self.fc_backward(self, self._lastLayer)
        else:
            self.image_backward(self, self._lastLayer)

    def updateWeight(self,speed,loss,):
        if self.trainable:
            for currLayercell in self.cells:
                currLayercell.updateWeight(speed, loss,self.clip_min,self.clip_max )

    def backPropagation(self,speed,loss):              #最后一个层调用,往前跑
        currLayer = self
        lastLayer = self._lastLayer
        while lastLayer:        #计算所有的error
            currLayer.backward()
            currLayer = lastLayer
            lastLayer = lastLayer._lastLayer
        while currLayer:            #更新权重
            currLayer.updateWeight(speed,loss)
            currLayer = currLayer._nextLayer

    ####################增加神经元##################################
    def addCell(self,currW,currH,nextW):
        cell = Cell(self.activation,self._specialCellType)
        self.activation.setWeightInitParameter( currW, -currH, False)
        cell.setInputCells(self._lastLayer.cells if hasattr(self._lastLayer, 'cells') else self._lastLayer.outImage)
        self.cells.append(cell)
        if self._nextLayer:
            for cell in self._nextLayer.cells:
                cell.w.append(nextW)
                cell.delta_w.append(0)
                if cell.specialCellType:
                    cell.h.append(0)
                    cell.delta_h.append(0)

    def setCellsStatisticEnable(self,enable = True):
        for cell in self.cells:
            if enable:
                cell.maxOutMultiWeight = 0          #清掉
                cell.totalOutMultiWeight = 0        #清掉
            cell.statisticEnable = enable

    def setCellsMinusWeightEnable(self,enable):       #使权重只朝向0的方向更新
        for cell in self.cells:
            cell.MinusWeightEnable = enable

    ###################减少神经元###############
    def tryToDeleteCell(self,randomFlag):
        minCellIndex = 0
        if randomFlag:
            minCellIndex = random.randint(0,len(self.cells) - 1)
        else:
            i = 0
            totalLossMin = -10000000
            for cell in self.cells:
                cell.statisticEnable = False
                if cell.totalOutMultiWeight > totalLossMin:
                    minCellIndex = i
                i += 1
        del self.cells[minCellIndex]
        if self._nextLayer:
            for cell in self._nextLayer.cells:
                del cell.w[minCellIndex]
                del cell.delta_w[minCellIndex]
                if cell.specialCellType:
                    del cell.h[minCellIndex]
                    del cell.delta_h[minCellIndex]

    def getLossStatistic(self,inputLayer,x,y):
        lossTotal = 0
        # result = []
        maxLossPos = None
        maxLossAbs = 0
        outputLayer = self._nextLayer
        if outputLayer:
            self.setCellsStatisticEnable(True)
            for t in range(len(x)):             #
                inputLayer.setInputAndForward(x[t])
                # result.append(outputLayer.cells[0].out)
                for cell in outputLayer.cells:
                    currLoss = (y[t] - cell.out)
                    currLossAbs = abs(currLoss)
                    if currLossAbs > maxLossAbs:
                        maxLossAbs = currLossAbs
                        maxLoss = currLoss
                        maxLossPos = x[t]
                    lossTotal += currLossAbs
            self.setCellsStatisticEnable(False)
        else:
            raise Exception("Next Layer should not be None")
        return (maxLoss,maxLossPos,lossTotal)

    def autoScheduleCells(self,lossLowerLimit, lossUpperLimit, timeoutUpperLimit, timeoutIncreaseCell, timeoutDecreaseCell,
                          maxLoss, maxLossPos, lossTotal,singleInputProbabilityNetwork):
        if not hasattr(self,'minimumLossFlag'):
            self.minimumLossFlag = False
            self.lastTotalLoss = lossTotal
            self.lastTime = time.time()
        if self.minimumLossFlag:
            if abs(maxLoss) > lossLowerLimit:
                timeDiff = time.time() - self.lastTime
                timeout = timeDiff > timeoutUpperLimit
                if lossTotal < self.lastTotalLoss * (len(self.cells) - 3) / len(self.cells) or timeout:
                    self.lastTime = time.time()
                    if timeout:
                        singleInputProbabilityNetwork.updateDistribution(False, 0.1)  #
                    else:
                        singleInputProbabilityNetwork.updateDistribution(True, 0.1)
                    self.lastTotalLoss = lossTotal
                    wCoe = singleInputProbabilityNetwork.getPredictiveOutput()
                    print("timeout wCoe:", wCoe, " timeDiff:", timeDiff)
                    if timeDiff > timeoutIncreaseCell:  # 仅当长时间lost不下降的情况下才增加神经元
                        self.addCell(wCoe, np.reshape(maxLossPos, (-1,)), maxLoss)
            else:
                self.minimumLossFlag = False
        else:
            if abs(maxLoss) < lossUpperLimit:
                timeDiff = time.time() - self.lastTime
                if timeDiff > timeoutDecreaseCell:
                    self.lastTime = time.time()
                    self.tryToDeleteCell(np.random.choice([True, False, False, False,False,False]))
            else:
                self.minimumLossFlag = True

class RnnLayer(FcLayer):
    def __init__(self,lastLayer = None,cellNum = 1,activation = None,specialCellType = False,trainable = True):
        self.recordValue = [0 for i in range(cellNum)]
        super().__init__(lastLayer = lastLayer,cellNum = cellNum,activation = activation,specialCellType = specialCellType,trainable = trainable)


    def setLastLayer(self, lastLayer):
        if lastLayer != None and not lastLayer._nextLayer:
            lastLayer._nextLayer = self
            self._lastLayer = lastLayer
            for cell in self.cells:
                cell.setInputCells(lastLayer.cells if hasattr(lastLayer,'cells') else lastLayer.outImage)
                cell.setExtraInput(self.recordValue)                                    ###########这个是必须的


    def updateWeight(self,speed,loss):
        if self.trainable:
            for currLayercell in self.cells:
                currLayercell.updateWeight(speed, loss,self.clip_min,self.clip_max)
        ##############################################################
        for i in range(len(self.cells)):
            # print(".....",self.cells[i].out  )
            self.recordValue[i] = self.cells[i].out         #将输出记录下来
        ##############################################################



class Conv2dLayer(Layer):
    def __init__(self,kernel_shape,lastLayer = None,imageShape = None,trainable = True,specialCore = False):         #仅在第一层的图像输入层输入imageShape
        super().__init__(trainable=trainable)
        self.specialCore = specialCore
        self._lastLayer = lastLayer
        self.kernel = np.random.random_sample(kernel_shape)
        (self.kernel_row,self.kernel_col) = kernel_shape
        (self.kernel_row_start, self.kernel_col_start) = (self.kernel_row//2,self.kernel_col//2)
        self._nextLayer = None;

        if lastLayer:
            self.setLastLayer(lastLayer)

        self.outImage = None
        if imageShape:
            self.outImage = np.zeros(imageShape, dtype=np.float32)


    def setLastLayer(self,lastLayer):
        if lastLayer == None:
            if self.outImage == None:
                raise Exception("first layer should input imageShape")
        else:
            self._lastLayer = lastLayer
            lastLayer._nextLayer = self
            image = lastLayer.outImage
            self.lastLayerImage = image
            (self.image_row, self.image_col) = image.shape
            self.outImage = np.zeros(image.shape,dtype=np.float32)
            self.error = np.zeros(image.shape, dtype=np.float32)
            self.image_rows, self.image_cols = self.outImage.shape


    def _forward(self):                      #第一个层调用
        nextLayer = self._nextLayer
        while nextLayer:
            nextLayer.caculateOut()
            nextLayer = nextLayer._nextLayer
    def setInputAndForward(self,x):           #仅第一层调用
        self.outImage[:,:] = x[:,:]
        self._forward()


    def _caculate_conv2d(self,x,y):         #卷积
        value = 0.
        for i in range(self.kernel_row):
            curr_y = y + i - self.kernel_row_start;
            if curr_y < 0 or curr_y >= self.image_rows:
                continue
            for j in range(self.kernel_col):
                curr_x = x + j - self.kernel_col_start;
                if curr_x < 0 or curr_x >= self.image_cols:
                    continue
                if self.specialCore:
                    diff = self.kernel[i][j] - self.lastLayerImage[curr_y][curr_x]
                    value += abs(diff)  #diff*diff;
                else:
                    value += self.kernel[i][j] * self.lastLayerImage[curr_y][curr_x];
        return value/self.outImage.size;

    def caculateOut(self):
        for y in range(self.image_rows):
            for x in range(self.image_cols):
                self.outImage[y][x] = max(self._caculate_conv2d(x,y),0.)             #relu


    def _updateKernel(self,x,y,speed,loss):
        value = 0.
        for i in range(self.image_rows):
            curr_y = i - (y - self.kernel_row_start);
            if curr_y < 0 or curr_y >= self.image_rows:
                continue
            for j in range(self.image_col):
                curr_x = j - (x  - self.kernel_col_start);
                if curr_x < 0 or curr_x >= self.image_cols:
                    continue

                if(self.outImage[i][j] <= 0.):       #relu
                    continue

                if self.specialCore:
                    diff = self.kernel[y][x] - self.lastLayerImage[curr_y][curr_x]
                    value += self.error[i][j] * (1 if diff > 0 else -1);      #2*self.error[i][j] * diff
                else:
                    value += self.error[i][j]*self.lastLayerImage[curr_y][curr_x];
        self.kernel[y][x] -= speed*value/self.outImage.size;

    def updateWeight(self,speed,loss):
        if self.trainable:
            for y in range(self.kernel_row):
                for x in range(self.kernel_col):
                    self._updateKernel(x,y,speed,loss)


    def _caculate_error(self,x,y):         #传播error
        value = 0.
        for i in range(self.kernel_row):
            curr_y = y - (i - self.kernel_row_start);
            if curr_y < 0 or curr_y >= self.image_rows:
                continue
            for j in range(self.kernel_col):
                curr_x = x - (j - self.kernel_col_start);
                if curr_x < 0 or curr_x >= self.image_cols:
                    continue

                if(self.outImage[curr_y][curr_x] <= 0.):       #relu
                    continue

                if self.specialCore:
                    diff = self.lastLayerImage[curr_y][curr_x] - self.kernel[i][j]
                    value += (1 if diff >= 0 else -1)*self.error[curr_y][curr_x]#2*diff * self.error[curr_y][curr_x];
                else:
                    value += self.kernel[i][j]*self.error[curr_y][curr_x];
        return value/self.outImage.size;

    def backward(self):
         if(hasattr(self._lastLayer,'error')):
             for y in range(self.image_rows):
                for x in range(self.image_cols):
                    self._lastLayer.error[y][x] = self._caculate_error(x,y)


class Pool(Layer):
    def __init__(self,pool_shape,lastLayer = None):         #仅在第一层的图像输入层输入imageShape
        super().__init__(trainable = False)
        self._lastLayer = lastLayer
        (self.pool_row,self.pool_col) = pool_shape
        self._nextLayer = None;
        if lastLayer:
            self.setLastLayer(lastLayer)


    def setLastLayer(self, lastLayer):
        if not lastLayer:
            raise Exception("Pool should not the first Layer")
        else:
            self._lastLayer = lastLayer
            lastLayer._nextLayer = self
            image = lastLayer.outImage
            self.lastLayerImage = image
            (self.image_row, self.image_col) = image.shape
            (self.new_row,self.new_col) = ((image.shape[0] + self.pool_row - 1)//self.pool_row,(image.shape[1] + self.pool_col - 1)//self.pool_col)
            self.outImage = np.zeros((self.new_row,self.new_col),dtype=np.float32)
            self.error = np.zeros(self.outImage.shape, dtype=np.float32)


    def updateWeight(self,speed,loss):
        pass            #pool不需要更新权重

class MaxPoolLayer(Pool):
    def caculateOut(self):
        self.outImageIndex = np.zeros((self.new_row, self.new_col), dtype=np.int)
        for i in range(self.new_row):
            for j in range(self.new_col):
                value = -100000000000.          #small enough
                for y in range(self.pool_row):
                    for x in range(self.pool_col):
                        new_x = x + j*self.pool_col
                        new_y = y + i*self.pool_row
                        if(new_x < self.image_col and new_y < self.image_row):
                            if self.lastLayerImage[new_y][new_x] > value:
                                self.outImageIndex[i][j] = (y << 16) + x
                                value = self.lastLayerImage[new_y][new_x]
                self.outImage[i][j] = value

    def backward(self):
        for i in range(self.new_row):
            for j in range(self.new_col):
                hitY = self.outImageIndex[i][j] >> 16
                hitX = self.outImageIndex[i][j] & 0xffff
                for y in range(self.pool_row):
                    for x in range(self.pool_col):
                        new_x = x + j*self.pool_col
                        new_y = y + i*self.pool_row
                        if (new_x < self.image_col and new_y < self.image_row):
                            if y == hitY and x == hitX:
                                self._lastLayer.error[new_y][new_x] = self.error[i][j]
                            else:
                                self._lastLayer.error[new_y][new_x] = 0

class AveragePool():
    def setInputImages(self):
        pass
    def caculateOut(self):
        pass
    def updateWeight(self,speed,loss):
        pass

class LayerStack():
    def __init__(self):
        self.layers = []
    def push(self,layer):
        lastLayer = None if not len(self.layers) else self.layers[-1]

        layer.setLastLayer(lastLayer)
        self.layers.append(layer)
        return layer


class Loss:
    def __init__(self, layer):
        self._layer = layer
        pass

    def minimize(self, expect):
        raise NotImplemented("")


class LossL2(Loss):
    def __init__(self, layer):
        super().__init__(layer)
        if (len(layer.cells) != 1):
            raise (Exception("last layer shoule only one cell!"))

    def minimize(self, expect, speed):  # L2距离为  （out - expect)^2   ,其偏导为 2*(out - expect)
        loss = (self._layer.cells[0].out - expect) * (self._layer.cells[0].out - expect)
        self._layer.cells[0].error = 2 * (self._layer.cells[0].out - expect)
        self._layer.backPropagation(speed, loss)

class LossLiner(Loss):
    def __init__(self, layer):
        super().__init__(layer)
        if (len(layer.cells) != 1):
            raise (Exception("last layer shoule only one cell!"))

    def minimize(self, loss, speed):
        loss = loss
        self._layer.cells[0].error_deri = 1#loss
        self._layer.backPropagation(speed, loss)

    def maximization(self, loss, speed):
        loss = loss
        self._layer.cells[0].error_deri = -1#loss
        self._layer.backPropagation(speed, loss)

class LossEntropy(Loss):  # 通常是配合前一级是 sigmoid函数的损失计算，否则意义不大
    def __init__(self, layer):
        super().__init__(layer)
        if (len(layer.cells) != 1):
            raise (Exception("last layer shoule only one cell!"))

    def minimize(self, expect,
                 speed):  # 距离为  -(expect*ln(out) + (1 - expect)*ln(1 - out)   ,其偏导为 -(expect/out - (1 - expect)/(1 - out)) = (out - expect)/((1 - out)*out) ，因为error有一个除法，很容易在计算的时候，数据超出浮点数范围
        loss = 0 #实际不需要计算该值，对后面无影响 ，并且计算该值可能会造成log值超过极限 -(expect * math.log(self._layer.cells[0].out) + (1 - expect) * math.log(1 - self._layer.cells[0].out))
        if(self._layer.activation.__class__ == ActivationSigmoid):
            self._layer.cells[0].error_deri = (self._layer.cells[0].out - expect)
        else:
            self._layer.cells[0].error = (self._layer.cells[0].out - expect) / ( self._layer.cells[0].out * (1 - self._layer.cells[0].out))
        self._layer.backPropagation(speed, loss)




import shutil
import  os

def runGAN():

    hideCellNum = 30        #隐含层神经元数目
    speed = 0.001         #不要小看这个speed,选择过大的时候，非常容易造成递度爆炸，比如你可以试试speed为1，Relu的训练
    WGAN = False

    #imageDir = "tempPic"
    #shutil.rmtree(imageDir, True)
    #os.mkdir(imageDir)

    layerStack = LayerStack()
    gInputLayer = layerStack.push(FcLayer(None,1,ActivationLiner(1,0),trainable=False))
    gHidelayer = layerStack.push(FcLayer(None,hideCellNum,ActivationRelu(1,8),trainable=True))
    gOutputLayer = layerStack.push(FcLayer(None, 1, ActivationLiner(1, 1), trainable=True))

    if WGAN:            #对WGAN理论讲得最好的中文解释，没有之一：https://zhuanlan.zhihu.com/p/25071913
        dHideLayer = layerStack.push(FcLayer(None,hideCellNum,ActivationRelu(0.1,5),trainable=True))
        dHideLayer.clip_by_value(-1,1)      #WAN一定要限制w值的范围
        dOutputLayer = layerStack.push(FcLayer(None, 1, ActivationLiner(1, 0), trainable=True))
        dOutputLayer.clip_by_value(-1, 1)
        lossD = LossLiner(dOutputLayer)
    else:
        dHideLayer = layerStack.push(FcLayer(None,hideCellNum,ActivationRelu(0.3,10),trainable=True))
        dOutputLayer = layerStack.push(FcLayer(None, 1, ActivationSigmoid(1, 0), trainable=True))
        lossD = LossEntropy(dOutputLayer)#LossEntropy(dOutputLayer)

    x = np.concatenate((np.linspace(-2, -1, 10),np.linspace(1, 3, 10)),axis=0)      #有效数据的范围

    g_i = np.linspace(-3,3,20)
    y_g = np.zeros_like(g_i)




    plt.close()  # clf() # 清图  cla() # 清坐标轴 close() # 关窗口

    plt.ion()  # interactive mode on

    plt.figure(1,figsize=(12, 8))  # 创建图表1

    ax1 = plt.subplot(111)  # 在图表2中创建子图1



    for loop in range(1000000):

        gHidelayer.setTrainable(False)
        gOutputLayer.setTrainable(False)        #训练判别器的时候，生成器的参数全部都不用动
        dHideLayer.setTrainable(True)
        dOutputLayer.setTrainable(True)
        #判别器
        for subLoop in range(1):
            for t in range(len(x)):             #真实的数据
                gOutputLayer.setInputAndForward([x[t]])     #[0]
                if WGAN:
                    lossD.maximization(dOutputLayer.cells[0].out, speed)
                else:
                    lossD.minimize(1.0, speed)
            for t in range(len(g_i)):
                gInputLayer.setInputAndForward([g_i[t]])    #生成器的数据
                if WGAN:
                    lossD.minimize(dOutputLayer.cells[0].out, speed)
                else:
                    lossD.minimize(0.0,speed)


        gHidelayer.setTrainable(True)
        gOutputLayer.setTrainable(True)
        dHideLayer.setTrainable(False)          #训练生成器的时候，判别器的参数不动
        dOutputLayer.setTrainable(False)
        # 生成器
        for subLoop in range(1):
            for t in range(len(g_i)):
                gInputLayer.setInputAndForward([g_i[t]])
                if WGAN:
                    lossD.maximization(dOutputLayer.cells[0].out, speed)
                else:
                    lossD.minimize(1.0,speed)


        if True:
            #plt.savefig(imageDir + '/' + str(loop))

            ax1.clear()
            ax1.grid(True)  # 添加网格
            ax1.set_title('WGAN:' + str(WGAN) + ' result loop:' + str(loop) + ' Cell:' + str(hideCellNum))  # 目标函数，补经网络的输出，以及隐含层每个神经元的输出乘以相应w权重


            for t in range(len(g_i)):
                gInputLayer.setInputAndForward([g_i[t]])        #生成器的输出
                y_g[t] = gOutputLayer.cells[0].out

            all_i = np.linspace(min(np.min(x),np.min(y_g)) - 1, max(np.max(x),np.max(y_g)) + 1, 60)
            y_all = np.zeros_like(all_i)
            for t in range(len(all_i)):  #
                gOutputLayer.setInputAndForward([all_i[t]])     #判别器的输出
                y_all[t] = dOutputLayer.cells[0].out

            ax1.plot(all_i, y_all,'-') #,
            ax1.plot(y_g, g_i , '-*')
            ax1.plot(x,np.ones_like(x),'-o')                    #真实数据的分布
            plt.pause(0.1)



if __name__ == "__main__":

    runGAN()