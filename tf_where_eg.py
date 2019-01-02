import tensorflow as tf
a = [[1,2,3],[4,5,6]]
b = [[1,0,3],[1,5,1]]
condition1 = [[True,False,False],
             [False,True,True]]
condition2 = [[True,False,False],
             [False,True,False]]
with tf.Session() as sess:
    print(sess.run(tf.where(condition1)))
    print(sess.run(tf.where(condition2)))
'''
[[0 0]
 [1 1]
 [1 2]]
 
[[0 0]
 [1 1]]
'''


'''返回值是对应元素，condition中元素为True的元素替换为x中的元素，为False的元素替换为y中对应元素'''
x = [[1,2,3],[4,5,6]]
y = [[7,8,9],[10,11,12]]
condition3 = [[True,False,False],
             [False,True,True]]
condition4 = [[True,False,False],
             [True,True,False]]
with tf.Session() as sess:
    print(sess.run(tf.where(condition3,x,y)))
    print(sess.run(tf.where(condition4,x,y)))

'''
[[ 1  8  9]
 [10  5  6]]
[[ 1  8  9]
 [ 4  5 12]]

'''
