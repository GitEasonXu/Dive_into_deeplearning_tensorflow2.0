
def backfoward(fun, x):
    delt_x = 0.0001
    return (fun(x + delt_x) - fun(x)) / delt_x

def update(x, delt):
    return x - 0.01*delt

def function(x):
    return (x - 3) ** 2 + 8           #函数的极值为3

#定义求极值函数  y=(x - 3)^2 + 8
func = function 
#x取一个初始值                     
init_x = 100
#已知最小值点
mix_point = 3
step = 0
for i in range(1000):
    delt = backfoward(func, init_x)  #计算梯度
    init_x = update(init_x, delt)    #跟新极值点
    if abs(init_x - mix_point) < 0.00001:
        step = i
        break
print("迭代 %d 轮，函数极小值点: %.10f"%(step, init_x))
