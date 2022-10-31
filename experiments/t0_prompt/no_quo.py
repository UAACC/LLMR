with open(r'/home/mrli/projects/def-lilimou/mrli/data/ost/T0_gen_results_ost.tgt','r') as fin:
    with open(r'/home/mrli/projects/def-lilimou/mrli/data/ost/T0_gen_results_ost_noquo.tgt','w') as fout:
        lines = fin.readlines()
        for i in lines:
            i = i.strip().strip('"')
            fout.writelines(i)
            fout.write('\n')