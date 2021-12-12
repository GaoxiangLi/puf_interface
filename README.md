# puf_interface

This is the source code of the paper "Challenge Input Interfaces for Arbiter PUF Variants Against Machine Learning Attacks"

Author: Yu Zhuang, Khalid Mursi, Gaoxiang Li

How to use:

1.Install required package:   

                        Python version 3.6
                        tensorflow 2.6
                        numpy 1.19.5
                        scikit-learn 0.24.2
                        pypuf 2.3.1
        
2.After download and unzip the file,  

                  Use your terminal to run the command:   python puf_interface.py   in this directory. 
                  
                  Use the command:  python puf_interface.py --help (for help to show all command line instruction)  

3.Parameter in command line 
  
                  -h, --help            show this help message and exit  
                  
                  --plus                Use this option for plus interface  
                  
                  --minus               Use this option for minus interface  
                  
                  --permutation         Use this option for permutation interface  
                  
                  --interface INTERFACE
                                        Use this option for applying interface  default = 1
                                        
                  --seed SEED           The seed for generating PUF instance  default = 0
                  
                  --batch_size BATCH_SIZE
                                        The batch size for training  default = 100000
                                        
                  --max_epoch MAX_EPOCH
                                        The max epoch for training  default = 500
                                        
                  --patience PATIENCE   The patience for training  default = 5
                  
                  --n n                 The number of stages of the generated PUF  default = 64
                  
                  --k K                 The number of challenges of the generated PUF  default = 1
                  
                  --N N                 The number of generated CPR for training  default = 10000000
                  
                  --noise NOISE         The noise level of generated PUF  default = 0.00

                  --ghost_bit_len GHOST_BIT_LEN
                                        The number of ghost bit for plus interface default = 1 

                  --double_use_bit_len DOUBLE_USE_BIT_LEN
                                        The number of ghost bit for minus interface  default = 1

                  --unsatationary_bit_len UNSATATIONARY_BIT_LEN
                                        The number of ghost bit for permutation interface  default = 3

                  --group GROUP         The number of groups for applied interface  default = 0
                  
                  --hop HOP             The hop setting for permutation interface  default = 2
                  
                  --puf PUF             Use this option to choose PUF type  default = 'apuf'

3.Experiment results output file path:   

                  Example pf output path: "ffpuf_without_interface.csv", or "3_64xpuf_interface_plus.csv"  
                  
                  Output path can be edited pr modified in each run() function             
             
4.Source code files include:  

                  puf_interface.py :dispatcher file  

                  interface_permuation.py  : for permuation interface method  

                  interface_minus.py  : for minus interface method  

                  interface_plus.py : for plus interface method  

                  Detail comments for each file and function are given in the code file  
                  
5. Command line example:
                  
                  python puf_interface.py --plus --interface --seed 0 --ghost_bit_len 10 --puf apuf  
                  
                  python puf_interface.py --minus --interface --seed 1 --k 3 --double_use_bit_len 5 --puf xpuf  
                  
                  python puf_interface.py --permutation --interface --seed 2 --k 1 --unsatationary_bit_len 10 --hop 3 --puf ffpuf  


  
 
  
