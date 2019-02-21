__init__.py	            first modify for copynet	a year ago
  attention_wrapper.py	  cp seq2seq files to local	a year ago
basic_decoder.py	    unstable code update	    a year ago
  decoder.py	              cp seq2seq files to local	a year ago
helper.py	            first modify for copynet	a year ago
  loss.py	                  cp seq2seq files to local


1.自己重写了TrainingHelper为CopyNetTrainingHelper
2.自己重写了BasicDecoder为CopyNetDecoder(BasicDecoder)，从encoder_outputs计算获得attention,后面step解码时进行重写