build/tools/caffe time -gpu all -model $@ -iterations 5 2>&1 | tee log.txt
