CUDA_VISIBLE_DEVICES=0 python main.py --data yelp --reg 1e-2 --temp 0.1 --ssl_reg 1e-7  --save_path yelp12 --epoch 150  --batch 512 --sslNum 40 --graphNum 12 --gnn_layer 3  --att_layer 2 --test True --testSize 1000 --ssldim 32 --sampNum 40