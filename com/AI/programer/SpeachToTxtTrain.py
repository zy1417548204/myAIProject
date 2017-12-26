import os;
import sys;
import pickle;
import librosa;
import numpy as np;
from keras.models import Model;
from keras import backend as K;
from keras.layers.embeddings import Embedding;
from keras.utils.vis_utils import plot_model;
from keras.models import Sequential, load_model;
from keras.optimizers import rmsprop, adam, adagrad, SGD;
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau;
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer;
from keras.layers import Input, Dense, merge, Dropout, BatchNormalization, Activation, Conv1D, Lambda;
#返回当前进程的工作目录
DIR=os.getcwd();

with open(DIR+"/train.word.txt") as f:
    #按行切割
    texts=f.read().split("\n");
#删除最后一行
del texts[-1];
#遍历每一行，并对其按空格切割
texts=[i.split(" ") for i in texts];
#定义一个空数组
all_words=[];
#
maxlen_char=0;
#从0到len
for i in np.arange(0,len(texts)):
    length=0;
    #遍历每一行的单词
    for j in texts[i][1:]:
        #每一行的单词的总长度
        length+=len(j);
    #如果长度不为空，maxlen_char=length;
    if maxlen_char<=length:maxlen_char=length;
    #将每一行所有单词都添加到数组中
    for j in np.arange(1,len(texts[i])):
        all_words.append(texts[i][j]);
"""
3 text.Tokenizer类

这个类用来对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示。 
init(num_words) 构造函数，传入词典的最大值

3.1 成员函数

fit_on_text(texts) 使用一系列文档来生成token词典，texts为list类，每个元素为一个文档。
texts_to_sequences(texts) 将多个文档转换为word下标的向量形式,shape为[len(texts)，len(text)] -- (文档数，每条文档的长度)
texts_to_matrix(texts) 将多个文档转换为矩阵表示,shape为[len(texts),num_words]
3.2 成员变量

document_count 处理的文档数量
word_index 一个dict，保存所有word对应的编号id，从1开始
word_counts 一个dict，保存每个word在所有文档中出现的次数
word_docs 一个dict，保存每个word出现的文档的数量
index_docs 一个dict，保存word的id出现的文档的数量

"""
tok=Tokenizer(char_level=True);

#生成token词典，每个元素为一个文档。
tok.fit_on_texts(all_words);
#获取文档索引
char_index=tok.word_index;

#value：key
index_char=dict((char_index[i],i) for i in char_index);
#10000行maxlen_char列的矩阵
char_vec=np.zeros((10000,maxlen_char),dtype=np.float32);
#char_input=[[] for _ in np.arange(0,len(texts))];

#表示行数之后
char_length=np.zeros((10000,1),dtype=np.float32);

#遍历这么len(texts)+1次
for i in np.arange(0,len(texts)):
    j=0;
    #遍历每一行的每个词
    for i1 in texts[i][1:]:
        #遍历每一行的每个字
        for ele in i1:
            char_vec[i,j]=char_index[ele];
            j+=1;
    char_length[i]=j;

'''mfcc_vec=[[] for _ in np.arange(0,len(texts))];
for i in np.arange(0,len(texts)):
    try:
        wav, sr = librosa.load(DIR + "/"+texts[i][0]+".wav", mono=True);
    except FileNotFoundError:
        wav, sr = librosa.load(DIR + "/" + texts[i][0] + ".WAV", mono=True);
    b = librosa.feature.mfcc(wav, sr)
    mfcc = np.transpose(b, [1, 0]);
    mfcc_vec[i]=mfcc;
    if i%100==0:print("Completed {}".format(str(i*len(texts)**-1)));
np.save(DIR+"/mfcc_vec",mfcc_vec);'''
'''mfcc_vec_origin=np.load(DIR+"/mfcc_vec_origin.npy");
maxlen_mfcc=673;
mfcc_vec=np.zeros((10000,maxlen_mfcc,20),dtype=np.float32);
for i in np.arange(0,len(mfcc_vec_origin)):
    for j in np.arange(0,len(mfcc_vec_origin[i])):
        for k,ele in enumerate(mfcc_vec_origin[i][j]):
            mfcc_vec[i,j,k]=ele;
np.save(DIR+"/mfcc_vec",mfcc_vec);'''

#加载数据
mfcc_input=np.load(DIR+"/mfcc_vec.npy");

#创建输入张量
input_tensor=Input(shape=(mfcc_input.shape[1],mfcc_input.shape[2]));
#构建网络模型
#Conv1D一维卷积 即（时域卷积），用以在一维输入信号上进行邻域滤波
"""
filters：卷积核的数目（即输出的维度）
kernel_size：整数或由单个整数构成的list/tuple，卷积核的空域或时域窗长度
strides：整数或由单个整数构成的list/tuple，为卷积的步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
padding：补0策略，为“valid”, “same” 或“causal”，“causal”将产生因果（膨胀的）卷积，即output[t]不依赖于input[t+1：]。当对不能违反时间顺序的时序信号建模时有用。参考WaveNet: A Generative Model for Raw Audio, section 2.1.。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
dilation_rate：整数或由单个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
use_bias:布尔值，是否使用偏置项
kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
kernel_regularizer：施加在权重上的正则项，为Regularizer对象
bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
activity_regularizer：施加在输出上的正则项，为Regularizer对象
kernel_constraints：施加在权重上的约束项，为Constraints对象
bias_constraints：施加在偏置上的约束项，为Constraints对象"""

#卷积核为1，个数为192，“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
x=Conv1D(kernel_size=1,filters=192,padding="same")(input_tensor);

#进行批标准化 该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1
"""
axis: 整数，指定要规范化的轴，通常为特征轴。例如在进行data_format="channels_first的2D卷积后，一般会设axis=1。
momentum: 动态均值的动量
epsilon：大于0的小浮点数，用于防止除0错误
center: 若设为True，将会将beta作为偏置加上去，否则忽略参数beta
scale: 若设为True，则会乘以gamma，否则不使用gamma。当下一层是线性的时，可以设False，因为scaling的操作将被下一层执行。
beta_initializer：beta权重的初始方法
gamma_initializer: gamma的初始化方法
moving_mean_initializer: 动态均值的初始化方法
moving_variance_initializer: 动态方差的初始化方法
beta_regularizer: 可选的beta正则
gamma_regularizer: 可选的gamma正则
beta_constraint: 可选的beta约束
gamma_constraint: 可选的gamma约束"""
#1）加速收敛 （2）控制过拟合，可以少用或不用Dropout和正则 （3）降低网络对初始化权重不敏感 （4）允许使用较大的学习率
#输入shape与输出shape一样
x=BatchNormalization(axis=-1)(x);
#激活函数，映射成-1到1之间到值
x=Activation("tanh")(x);

#x输入数据，size时域窗长度，rate,dim,卷积核个数，rate膨胀比例：空洞卷积
def res_block(x,size,rate,dim=192):
    x_tanh=Conv1D(kernel_size=size,filters=dim,dilation_rate=rate,padding="same")(x);
    x_tanh=BatchNormalization(axis=-1)(x_tanh);
    x_tanh=Activation("tanh")(x_tanh);

    x_sigmoid=Conv1D(kernel_size=size,filters=dim,dilation_rate=rate,padding="same")(x);
    x_sigmoid=BatchNormalization(axis=-1)(x_sigmoid);
    x_sigmoid=Activation("sigmoid")(x_sigmoid);

    #Merge层提供了一系列用于融合两个层或两个张量的层对象和方法。
    out=merge([x_tanh,x_sigmoid],mode="mul");#乘积
    """
    参数： 
layers：该参数为Keras张量的列表，或Keras层对象的列表。该列表的元素数目必须大于1。 
mode：合并模式，为预定义合并模式名的字符串或lambda函数或普通函数，如果为lambda函数或普通函数，则该函数必须接受一个张量的list作为输入，并返回一个张量。如果为字符串，则必须是下列值之一： 
“sum”，“mul”，“concat”，“ave”，“cos”，“dot” 
concat_axis：整数，当mode=concat时指定需要串联的轴 
dot_axes：整数或整数tuple，当mode=dot时，指定要消去的轴 
output_shape：整数tuple或lambda函数/普通函数（当mode为函数时）。如果output_shape是函数时，该函数的输入值应为一一对应于输入shape的list，并返回输出张量的shape。 
node_indices：可选，为整数list，如果有些层具有多个输出节点（node）的话，该参数可以指定需要merge的那些节点的下标。如果没有提供，该参数的默认值为全0向量，即合并输入层0号节点的输出值。 
tensor_indices：可选，为整数list，如果有些层返回多个输出张量的话，该参数用以指定需要合并的那些张量。 """
    out=Conv1D(kernel_size=1,filters=dim,padding="same")(out);
    out=BatchNormalization(axis=-1)(out);
    out=Activation("tanh")(out);
    #“cos” 余弦乘积
    #返回它们的和concat 拼接：返回它们的按照给定轴相接构成的向量
    #“dot” C = DOT(A,B) 返回向量A和B的内积，A,B必须长度相等。如果A,B都是列向量，DOT(A,B)=A'*B。
    #原始数据与合并层数据的乘积
    x=merge([x,out],mode="sum");
    return x,out;

skip=[];
for i in np.arange(0,3):
    for r in [1,2,4,8,16]:
        x,s=res_block(x,size=7,rate=r);
        #输出加入到数组中
        skip.append(s);
#自定义实现实现层函数
def ctc_lambda_function(args):
    y_true_input, logit, logit_length_input, y_true_length_input=args;
    #在batch上运行CTC损失算法"" y_true：形如(samples，max_tring_length)的张量，包含标签的真值 y_pred：形如(samples，time_steps，num_categories)的张量，包含预测值或输出的softmax值
    # input_length：形如(samples，1)的张量，包含y_pred中每个batch的序列长   label_length：形如(samples，1)的张量，包含y_true中每个batch的序列长
    return K.ctc_batch_cost(y_true_input,logit,logit_length_input,y_true_length_input);
#求和
skip_tensor=merge([s for s in skip],mode="sum");

logit=Conv1D(kernel_size=1,filters=192,padding="same")(skip_tensor);
logit=BatchNormalization(axis=-1)(logit);
logit=Activation("tanh")(logit);

#卷集核个数char_index)+1，所有的单词个数
logit=Conv1D(kernel_size=1,filters=len(char_index)+1,padding="same",activation="softmax")(logit);
#base_model=Model(inputs=input_tensor,outputs=logit);
logit_length_input=Input(shape=(1,));

#maxlen_char 总单词长度
y_true_input=Input(shape=(maxlen_char,));
y_true_length_input=Input(shape=(1,));

#Lambda
loss_out=Lambda(ctc_lambda_function,output_shape=(1,),name="ctc")([y_true_input,logit,logit_length_input,y_true_length_input])
#调用Model，类似于之前的
model=Model(inputs=[input_tensor,logit_length_input,y_true_input,y_true_length_input],outputs=loss_out);
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},optimizer="adam");
#plot_model(model, to_file="model.png", show_shapes=True);

#当检测数量停止提高时停止训练
"""
EarlyStopping的参数有

monitor: 监控的数据接口，有’acc’,’val_acc’,’loss’,’val_loss’等等。正常情况下如果有验证集，就用’val_acc’或者’val_loss’。但是因为笔者用的是5折交叉验证，没有单设验证集，所以只能用’acc’了。
min_delta：增大或减小的阈值，只有大于这个部分才算作improvement。这个值的大小取决于monitor，也反映了你的容忍程度。例如笔者的monitor是’acc’，同时其变化范围在70%-90%之间，所以对于小于0.01%的变化不关心。加上观察到训练过程中存在抖动的情况（即先下降后上升），所以适当增大容忍程度，最终设为0.003%。
patience：能够容忍多少个epoch内都没有improvement。这个设置其实是在抖动和真正的准确率下降之间做tradeoff。如果patience设的大，那么最终得到的准确率要略低于模型可以达到的最高准确率。如果patience设的小，那么模型很可能在前期抖动，还在全图搜索的阶段就停止了，准确率一般很差。patience的大小和learning rate直接相关。在learning rate设定的情况下，前期先训练几次观察抖动的epoch number，比其稍大些设置patience。在learning rate变化的情况下，建议要略小于最大的抖动epoch number。笔者在引入EarlyStopping之前就已经得到可以接受的结果了，EarlyStopping算是锦上添花，所以patience设的比较高，设为抖动epoch number的最大值。
mode: 就’auto’, ‘min’, ‘,max’三个可能。如果知道是要上升还是下降，建议设置一下。笔者的monitor是’acc’，所以mode=’max’。
"""
early = EarlyStopping(monitor="loss", mode="min", patience=10);

"""当指标变化小时，减少学习率 当学习停滞时，减少2倍或10倍的学习率常常能获得较好的效果。该回调函数检测指标的情况，
如果在patience个epoch中看不到模型性能提升，则减少学习率
monitor：被监测的量
factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
epsilon：阈值，用来确定是否进入检测值的“平原区”
cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
min_lr：学习率的下限
"""
lr_change = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=0, min_lr=0.000)
checkpoint = ModelCheckpoint(filepath=DIR + "/listen_model.chk",
                              save_best_only=False);
#适应性矩估计
"""
Adam 通过计算梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应性学习率。
alpha：同样也称为学习率或步长因子，它控制了权重的更新比率（如 0.001）。较大的值
（如 0.3）在学习率更新前会有更快的初始学习，而较小的值（如 1.0E-5）会令训练收敛
到更好的性能。

beta1：一阶矩估计的指数衰减率（如 0.9）。

beta2：二阶矩估计的指数衰减率（如 0.999）。该超参数在稀疏梯度（如在 NLP 或计算机视觉任务中）
中应该设置为接近 1 的数。

epsilon：该参数是非常小的数，其为了防止在实现中除以零（如 10E-8）。

Adam 论文建议的参数设定：
测试机器学习问题比较好的默认参数设定为：alpha=0.001、beta1=0.9、beta2=0.999 和 epsilon=10E−8。
我们也可以看到流行的深度学习库都采用了该论文推荐的参数作为默认设定。
TensorFlow：learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08.
Keras：lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0.
Blocks：learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-08, decay_factor=1.
Lasagne：learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08
Caffe：learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08
MxNet：learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
Torch：learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
在第一部分中，我们讨论了 Adam 优化算法在深度学习中的基本特性和原理：
Adam 是一种在深度学习模型中用来替代随机梯度下降的优化算法。
Adam 结合了 AdaGrad 和 RMSProp 算法最优的性能，它还是能提供解决稀疏梯度和噪声问题的优化方法。
Adam 的调参相对简单，默认参数就可以处理绝大部分的问题。
而接下来的第二部分我们可以从原论文出发具体展开 Adam 算法的过程和更新规则等。"""
opt=adam(lr=0.0003);
model.fit(x=[mfcc_input,np.ones(10000)*673,char_vec,char_length],y=np.ones(10000),callbacks=[early,lr_change,checkpoint],
          batch_size=50,epochs=1000);