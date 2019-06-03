from mxnet.gluon import nn
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, \
    BatchNorm, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout

class self_attention_block(nn.HybridBlock):
    def __init__(self,in_channel):
        super(self_attention_block, self).__init__()
        self.in_channel=in_channel

        self.gp = nn.GlobalAvgPool2D()
        self.conv1=nn.Conv2D(in_channels=self.in_channel,channels=self.in_channel//16,kernel_size=1,strides=1,use_bias=False,activation='relu')
        self.conv2=nn.Conv2D(in_channels=self.in_channel//16,channels=self.in_channel,kernel_size=1,strides=1,use_bias=False,activation='sigmoid')
    def hybrid_forward(self, F, x):
        y=self.gp(x)
        y=self.conv1(y)
        y=self.conv2(y)
        x=F.broadcast_mul(x,y)
        return x

class CA_M(nn.HybridBlock):
    def __init__(self):
        super(CA_M,self).__init__()
        self.gp = nn.GlobalAvgPool2D()
    def hybrid_forward(self, F, x):
        x1=x.reshape((0,0,-1))
        gpx=self.gp(x)
        gpx1=gpx.reshape((0,0,1))
        gpx2=gpx1.transpose((0,2,1))
        rela_x=F.batch_dot(gpx1,gpx2)
        rela_x= F.softmax(rela_x,axis=-1)
        att_x=F.batch_dot(rela_x,x1)
        y=F.reshape_like(att_x, x)
        return y+x
    
class CA_M1(nn.HybridBlock):
    def __init__(self):
        super(CA_M1,self).__init__()
        self.gp = nn.GlobalAvgPool2D()
        self.bn=nn.BatchNorm()
    def hybrid_forward(self, F, x):
        x1=x.reshape((0,0,-1))
        
        num=x1.shape[2]
        
        x1_m=self.gp(x).reshape((0,0,1))
        
        cov1=F.batch_dot(x1-x1_m,(x1-x1_m).transpose((0,2,1)))/num
        rela=F.softmax(cov1,axis=-1)
        att_x=F.batch_dot(rela,x1)
        y=F.reshape_like(att_x,x)
        out=self.bn(x+y)
        return out
#position attention
class CA_M2(nn.HybridBlock):
    def __init__(self,in_channel):
        super(CA_M2,self).__init__()
        self.in_channel=in_channel
        self.middel_channel=self.in_channel//2
        self.conv1=nn.Conv2D(in_channels=self.in_channel,channels=self.middel_channel,kernel_size=1,strides=1)
        self.bn1=nn.BatchNorm()
        
        self.conv2=nn.Conv2D(in_channels=self.in_channel,channels=self.middel_channel,kernel_size=1,strides=1)
        self.bn2=nn.BatchNorm()

        
        self.conv3=nn.Conv2D(in_channels=self.in_channel,channels=self.middel_channel,kernel_size=1,strides=1)
        self.bn3=nn.BatchNorm()
        
        self.conv_out=nn.Conv2D(in_channels=self.middel_channel,channels=self.in_channel,kernel_size=1,strides=1)
        
    def hybrid_forward(self, F, x):
        XA=self.conv1(x)
        XA=self.bn1(XA)
        X1=XA
        
        XB=self.conv2(x)
        XB=self.bn2(XB)

        
        XC=self.conv3(x)
        XC=self.bn3(XC)

        
        XA=XA.reshape((0,0,-1))
        XA=XA.transpose((0,2,1))#B*HW*C
        
        XB=XB.reshape((0,0,-1))#B*C*HW
        XC=XC.reshape((0,0,-1))#B*C*HW
        XC=XC.transpose((0,2,1))#B*HW*C
        att=F.batch_dot(XA,XB)
        
        att=F.softmax(att,axis=-1)
        
        X_ATT=F.batch_dot(att,XC)
        X_ATT=X_ATT.transpose((0,2,1))
        
        X_ATT=F.reshape_like(X_ATT,X1)
        X_ATT=self.conv_out(X_ATT)
        return X_ATT+x

class CA_M3(nn.HybridBlock):
    def __init__(self):
        super(CA_M3,self).__init__()
    def hybrid_forward(self, F, x):
        return x