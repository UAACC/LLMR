import re
import matplotlib.pyplot as plt
import numpy as np   
plt.scatter( np.random.rand(100), np.random.rand(100) )
plt.xscale( 'log' )  # You can include one of these two
plt.yscale( 'log' )  # lines, or both, or neither.
plt.show()


output = r'/home/mrli/projects/def-lilimou/mrli/logs/output-45165500.log'
outpu_result = r'/home/mrli/projects/def-lilimou/mrli/projects/LLMR/experiments/T5_t0p_CE/train_log'
graph = r'/home/mrli/projects/def-lilimou/mrli/projects/LLMR/experiments/T5_t0p_CE/train_loss.png'

def load_from_out(dir_in,dir_out):
    with open(dir_in,'r') as f:
        with open(dir_out,'w') as f_o:
            for line in f.readlines():
                if re.search(r'\bloss\b',line):
                    f_o.write(line)
                    # f_o.write('\n')

def draw_graph(dir_in,dir_out):
    reg_loss = r'(loss: (\w.\w*))'
    reg_u = r'(u: (\w*))'
    reg_gradn = r'(grad_norm: (\w.\w*))'
    train_updates=[]
    train_loss =[]
    grad_norm = []
    with open(dir_in,'r') as f:
        for i in f.readlines():
            updates=re.search(reg_u,i).group(2)
            loss = re.search(reg_loss,i).group(2)
            norm = re.search(reg_gradn,i).group(2)
            # norm = float(norm)*100.0
            # updates = int(updates)/100
            train_updates.append(updates)
            train_loss.append(loss)
            grad_norm.append(norm)

        # plt.xticks( range(1,500,25) )
        # plt.yticks( range(0,30,25) )
        plt.scatter( np.random.rand(100), np.random.rand(100) )
        plt.xscale( 'log' )  # You can include one of these two
        plt.yscale( 'log' )  # lines, or both, or neither.
        



        plt.plot(train_updates,train_loss,label = 'train_loss')
        plt.plot(train_updates,grad_norm,label = 'grad_norm')
        plt.xlabel('train_updates')
        plt.ylabel('figures')
        plt.title('Train monitor')
        plt.savefig(dir_out)


            
draw_graph(outpu_result,graph)









