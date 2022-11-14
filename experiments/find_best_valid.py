import re

valid_log = r'/home/mrli/scratch/projects/LLMR/ckpts/t5b-t0p-ost/valid.ost.log'

loc = r'/home/mrli/scratch/projects/LLMR/ckpts/t5b-t0p-ost/results.txt'


with open(valid_log) as f:
    L = []
    L2 = []
    L4 = []
    L_sum = []
    list_of_chars = ['B', 'L', 'E', 'U', '[',']','\n',' ']
    for line in f.readlines():
        if re.search(r'\bBLEU\b',line):
            for character in list_of_chars:
                line = line.replace(character, '')

            line = line.split(',')
            L.append(line)
    for i in L:
        two = float(i[1])
        four = float(i[3])
        sum_ =  two + four
        L2.append(two)
        L4.append(four)
        L_sum.append(sum_)
    
    max_2 = max(L2)
    max_2_model = L2.index(max_2) + 1
    
    max_4 = max(L4)
    max_4_model = L4.index(max_4) + 1

    max_sum = max(L_sum)
    max_sum_model = L_sum.index(max_sum) + 1
    max_sum_2 = L2[L_sum.index(max_sum)]
    max_sum_4 = L4[L_sum.index(max_sum)]

a = f'The best model in BLEU2 is model-{max_2_model}k at value of {max_2} and its BLEU4 is {L4[L2.index(max_2)]} sum at {L4[L2.index(max_2)] + max_2}\n'
b = f'The best model in BLEU4 is model-{max_4_model}k at value of {max_4} and its BLEU2 is {L2[L4.index(max_4)]} sum at {L2[L4.index(max_4)] + max_4}\n'
c = f'The best model in sum of BLEU2 and BLEU4 is model-{max_sum_model}k at value of {max_sum} and its BLEU2 is {max_sum_2} and BLEU4 is {max_sum_4}\n'

print(a)
print(b)
print(c)



with open(loc, 'w') as f:
    f.write(a + b + c)
    




