from tensorflow.contrib.seq2seq import BahdanauAttention,AttentionWrapper,BasicDecoder,dynamic_decode,sequence_loss
class PointGeneratorAttention(BahdanauAttention):
    def __init__(self):
        #子类初始化时先调用父类的初始化方法
        super(PointGeneratorAttention,self).__init__()
        pass
class PointGeneratorAttentionWrapper(AttentionWrapper):
    def __init__(self):
        super(PointGeneratorAttentionWrapper,self).__init__()
        pass
class PointGeneratorAttentionBasicDecoder(BasicDecoder):
    def __init__(self):
        super(PointGeneratorAttentionBasicDecoder, self).__init__()
