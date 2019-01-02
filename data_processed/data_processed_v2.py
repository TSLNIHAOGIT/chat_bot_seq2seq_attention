import pandas as pd
import jieba
import re
#添加自定义词典

#自定义词典要放在并行化程序之前，否则不起作用
jieba.load_userdict('self_define_dict.txt')

jieba.enable_parallel()


path = '/Users/ozintel/Downloads/Tsl_python_progect/local_ml/seq2seq-chinese-textsum/news/lcsts_part3.parquet.gzip'
'''
特殊字符：去除特殊字符，如：“「，」,￥,…”；
X:括号内的内容：如表情符，【嘻嘻】，【哈哈】 
？:日期：替换日期标签为TAG_DATE，如：***年*月*日，****年*月，等等
超链接URL：替换为标签TAG_URL；
删除全角的英文：替换为标签TAG_NAME_EN；#所有标点符号都被删除
替换数字：TAG_NUMBER；
'''



#添加停用词词典
with open('stopwords.txt') as stopwords:
    stopwords_set=set()
    for each in stopwords:
        stopwords_set.add(each.strip())
print('stopwords_set',stopwords_set)


TAG_DATE_PATTERN='\d{1,}年\d{1,}月\d{1,}日|\d{1,}年\d{1,}月|\d{1,}月\d{1,}日|\d{1,}年|\d{1,}月|\d{1,}日'
TAG_NUMBER_PATTERN='\d{1,}'
TAG_URL_PATTERN='[a-zA-z]+://[^\s]*'




counts = 0
def cut_process(input_str):
    global counts
    input_str=re.sub(TAG_DATE_PATTERN,'TAG_DATE',input_str)
    input_str=re.sub(TAG_NUMBER_PATTERN,'TAG_NUMBER',input_str)
    input_str=re.sub(TAG_URL_PATTERN,'TAG_URL',input_str)


    str_split=jieba.cut(input_str)
    cleaned_str_split=[each for each in str_split if each not in stopwords_set]

    if counts<3:
        counts=counts+1
        print('input', input_str)
        print('cleaned_str_split',cleaned_str_split)
    return ' '.join(cleaned_str_split)


def pre_sub_split_words(df):
    '''先分词将一些实体替换掉，例如日期时间、人名、地名、机构名字,还是先用规则处理'''
    # 1.去除特殊字符，直接去处所有的标点符号
    print('dataframe.info',df.info())
    df['content_split']=df.apply(lambda row:cut_process(row['content']),axis=1)
    df['title_split'] = df.apply(lambda row: cut_process(row['title']), axis=1)
    print(df[['content_split','title_split']].head(10))
    # df[['content_split','title_split']].to_csv('part2_split.csv')
    print('开始保存分好的词')
    df[['content_split', 'title_split']].to_parquet('part3_split.parquet.gzip',compression='gzip')




if __name__=='__main__':
    # path0='part1_split.parquet.gzip'
    data_df = pd.read_parquet(path)
    # pre_sub_split_words(data_df)
    pass
    print(data_df.head(5))
    '''
      content_split                                      title_split
0  新华社 受权 于 TAG_DATE 全文 播发 修改 后 的 中华人民共和国 立法法 修改 ...                                 修改 后 的 立法法 全文 公布
1  一辆 小轿车 一名 女司机 竟 造成 TAG_NUMBER 死 TAG_NUMBER 伤 日...   深圳机场 TAG_NUMBER 死 TAG_NUMBER 伤续 司机 全责 赔偿 或超 千万
2  TAG_DATE 习近平 总书记 对 政法 工作 作出 重要 指示 TAG_DATE 政法 ...             孟建柱 主动 适应 形势 新 变化 提高 政法 机关 服务大局 的 能力
3  针对 央视 TAG_NUMBER TAG_NUMBER 晚会 曝光 的 电信 行业 乱象 工...                           工信部 约 谈三大 运营商 严查 通信 违规
4  国家 食药监 管 总局 近日 发布 食品 召回 管理 办法 明确 食用 后 已经 或 可能 ...  食品 一级 召回 限 TAG_NUMBER 小时 内 启动 TAG_NUMBER 工作日 完成

    
    
    
    
    
    
    [['新华社受权于18日全文播发修改后的《中华人民共和国立法法》，修改后的立法法分为“总则”“法律”“行政法规”“地方性法规、自治条例和单行条例、规章”“适用与备案审查”“附则”等6章，共计105条。'
  '修改后的立法法全文公布']
 #['一辆小轿车，一名女司机，竟造成9死24伤。日前，深圳市交警局对事故进行通报：从目前证据看，事故系司机超速行驶且操作不当导致。目前24名伤员已有6名治愈出院，其余正接受治疗，预计事故赔偿费或超一千万元。'
  '深圳机场9死24伤续：司机全责赔偿或超千万']
 ['1月18日，习近平总书记对政法工作作出重要指示：2014年，政法战线各项工作特别是改革工作取得新成效。新形势下，希望全国政法机关主动适应新形势，为公正司法和提高执法司法公信力提供有力制度保障。'
  '孟建柱：主动适应形势新变化提高政法机关服务大局的能力']
 ['针对央视3·15晚会曝光的电信行业乱象，工信部在公告中表示，将严查央视3·15晚会曝光通信违规违法行为。工信部称，已约谈三大运营商有关负责人，并连夜责成三大运营商和所在省通信管理局进行调查，依法依规严肃处理。'
  '工信部约谈三大运营商严查通信违规']
 ['国家食药监管总局近日发布《食品召回管理办法》，明确：食用后已经或可能导致严重健康损害甚至死亡的，属一级召回，食品生产者应在知悉食品安全风险后24小时内启动召回，且自公告发布之日起10个工作日内完成召回。'
  '食品一级召回限24小时内启动10工作日完成']]
    '''