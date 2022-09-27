
#GCN
python main.py --dataset=Cora --type_model=GCN --alpha=1 --beta=10 --dropout=0.6 --lr=0.01 --epoch=1000 --cuda_num=0
python main.py --dataset=Citeseer --type_model=GCN --alpha=0.1 --beta=10 --dropout=0.6 --lr=0.01 --epoch=1000 --cuda_num=0
python main.py --dataset=Pubmed --type_model=GCN --alpha=1 --beta=10 --dropout=0.6 --lr=0.01 --epoch=1000 --cuda_num=0
python main.py --dataset=CoauthorCS --type_model=GCN --alpha=1 --beta=10 --dropout=0.6 --lr=0.01 --epoch=1000 --cuda_num=0 

#GAT
python main.py --dataset=Cora --type_model=GAT --alpha=1 --beta=1 --dropout=0.6 --lr=0.01 --epoch=1000 --cuda_num=0
python main.py --dataset=Citeseer --type_model=GAT --alpha=1 --beta=10 --dropout=0.6 --lr=0.01 --epoch=1000 --cuda_num=0
python main.py --dataset=Pubmed --type_model=GAT --alpha=1 --beta=1 --dropout=0.6 --lr=0.01 --epoch=1000 --cuda_num=0
python main.py --dataset=CoauthorCS --type_model=GAT --alpha=1 --beta=1 --dropout=0.6 --lr=0.01 --epoch=1000 --cuda_num=0

#Cheby
python main.py --dataset=Cora --type_model=Cheby --alpha=1 --beta=10 --dropout=0.6 --lr=0.01 --epoch=1000 --cuda_num=0
python main.py --dataset=Citeseer --type_model=Cheby --alpha=1 --beta=10 --dropout=0.6 --lr=0.01 --epoch=1000 --cuda_num=0 # For layer 15 
# python main.py --dataset=Citeseer --type_model=Cheby --alpha=0 --beta=10 --dropout=0.6 --lr=0.01 --epoch=1000 --cuda_num=0 # For layer 30 
python main.py --dataset=Pubmed --type_model=Cheby --alpha=1 --beta=10 --dropout=0.6 --lr=0.01 --epoch=1000 --cuda_num=0
python main.py --dataset=CoauthorCS --type_model=Cheby --alpha=1 --beta=10 --dropout=0.6 --lr=0.01 --epoch=1000 --cuda_num=0


