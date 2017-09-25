# some init in here

learning_rate = 0.0001  # Binary Network更新太慢，所以这个就是不能太大，要不直接就一下跳的值很大
iteration_numbers = 1000*10*60      # 按照paper里面是60 epochs，我因为没找到测试集的label--好笨 所以用训练集的每个前1000张训练，剩下300测试
epochs_number = 1000*10     # 1 epochs*classes_numbers
batch_size = 16
display_step = 1000
input_image = [84, 84, 3]  # weight, height
classes_name = ['n01440764', 'n01518878', 'n01614925', 'n01817953', 'n02835271', 'n03065424', 'n03724870', 'n04008634', 'n04009552', 'n10565667']   # imagenet中随机取得几组，没什么意思
classes_numbers = len(classes_name)
cwd = "G:\\ImageNet\\Images\\training\\"

parameters = []     # 存储所有变量
