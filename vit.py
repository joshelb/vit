import torch 
import torchvision
import os
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize ,Lambda
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler
import ray
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig




from datasets import load_dataset, load_from_disk
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def imgtopatch(x, patch_size):

    B, C, H, W = x.shape
    #print(x.shape)
    x = x.unfold(2, patch_size, patch_size)
    #print(x.shape)  
    x = x.unfold(3, patch_size, patch_size)  
    #print(x.shape)


    x = x.permute(0, 2, 3, 1, 4, 5)  
    #print(x.shape)      
    x = x.reshape(B, -1, C * patch_size * patch_size)  
    #print(x.shape)  

    return x

transform = Compose([
    #Lambda(lambda x: x.convert("RGB")),
    #Resize((256, 256)),  
    ToTensor(),          
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def transform_batch(batch):
    return {"image": [transform(img) for img in batch["image"]]}


#selected_classes = [3, 17, 42, 56, 100, 205, 333, 444, 555, 666]
#class_map = {k: i for i, k in enumerate(selected_classes)}

#def filter_fn(example):
#    return example['label'] in selected_classes
#def remap_labels(example):
#    example['label'] = class_map[example['label']]
#    return example

#filtered_ds = ds_train.filter(filter_fn)
#filtered_ds = filtered_ds.map(remap_labels)


def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location='cuda') 
    model.module.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm([embed_dim ])
        self.layer_norm2 = torch.nn.LayerNorm([embed_dim ])
        self.multihead = torch.nn.MultiheadAttention(embed_dim,12,batch_first=True)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.mlp = torch.nn.Sequential(
          torch.nn.Linear(embed_dim, 3072),
          torch.nn.GELU(),
          torch.nn.Dropout(p=0.1),
          torch.nn.Linear(3072, embed_dim)
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        attn_block, _ = self.multihead(x_norm, x_norm, x_norm)
        x = x + attn_block
        x_norm2 = self.layer_norm2(x)
        x = x + self.mlp(x_norm2)
        x = self.dropout(x)
        return x


class VisionTransformer(torch.nn.Module):

    def __init__(self,patchsize, num_patches, embed_dim, num_class):
        super(VisionTransformer, self).__init__()
        self.patchsize = patchsize
        self.linear_projection = torch.nn.Linear(patchsize * patchsize * 3,embed_dim)
        self.headclass = torch.nn.Parameter(torch.zeros(1,1,embed_dim)) 
        self.position_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) 
        self.dropout = torch.nn.Dropout(p=0.1)
        self.mlp_head = torch.nn.Sequential(
          torch.nn.Linear(embed_dim, 3072),
          torch.nn.GELU(),
          torch.nn.Dropout(p=0.1),
          torch.nn.Linear(3072, num_class)
        )
        self.encoder_layers = torch.nn.ModuleList([
            TransformerEncoderLayer(embed_dim, 12, 3072)
            for _ in range(12)
        ])
        

    def forward(self, x):
        B = x.shape[0]
        flattened_patches = imgtopatch(x,self.patchsize)
        x = self.linear_projection(flattened_patches)
        x = self.dropout(x)
        classtoken = self.headclass.expand(B,-1,-1)
        x = torch.cat((classtoken,x),dim=1)
        N = x.shape[1]
        D = x.shape[2]
        x = x + self.position_embed
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x)
        class_token = x[:, 0, :]
        x = self.mlp_head(class_token)
        #print(torch.argmax(x))
        return x


def setup(rank, world_size):
    #dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    #torch.cuda.set_device(rank % torch.cuda.device_count())


def cleanup():
    dist.destroy_process_group()

def train(config):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    #ds = load_dataset("imagenet-1k",cache_dir="/mnt/data/huggingface",data_dir="/mnt/data")
  

    #ds_test = test_ds.map(
      #  transform_batch,
     #   batched=True,
      ##  batch_size=2,
       # num_proc=10,
    #)
    ds_train = load_dataset("joshelb/imagenet1kvit",data_dir="imagenet_preprocessed", cache_dir="/data/huggingface", num_proc=8,split="train")
    ds_test = load_dataset("joshelb/imagenet1kvit",data_dir="imagenet_preprocessed_validation",cache_dir="/data/huggingface", num_proc=8,split="train")
    ds_train = ds_train.with_format("torch")
    ds_test = ds_test.with_format("torch")
    train_sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank,  shuffle=True)
    test_sampler = DistributedSampler(ds_test, num_replicas=world_size, rank=rank,  shuffle=True)



    train_dataloader = DataLoader(ds_train, batch_size=350,  sampler=train_sampler, num_workers=4 ,pin_memory=True)
    test_dataloader = DataLoader(ds_test, batch_size=350,  sampler=test_sampler, num_workers=4 ,pin_memory=True)

    num_epochs = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = VisionTransformer(16,256,768,1000).to(device)
    model = DDP(model, device_ids=[rank])
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    accumulation_steps = 2
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-3, betas=(0.9, 0.999), weight_decay=0.3)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=10000, total_steps=100000)
    
    #scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    #model, optimizer, epoch, loss = load_checkpoint('/home/ubuntu/vit/checkpoints/checkpoint_epoch_11.pt', model, optimizer)
    model.train()
    stepp = 0
    validation_stepp = 0
    for _ in range(int(stepp/accumulation_steps)):
        scheduler.step()
    writer = SummaryWriter(log_dir="/data/runs/vit_debug")
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        model.train() 
        epoch_loss = 0.0
        num_batches = 0
        print(epoch, stepp)
        for batch in train_dataloader:
            images,labels = batch["image"].to(device), batch["label"].to(device)
            output  = model(images)
            loss = criterion(output, labels)
            loss = loss / accumulation_steps
            writer.add_scalar('loss', loss.item() * accumulation_steps, stepp)
            loss.backward()    
            stepp += 1 
            if stepp % accumulation_steps == 0 and stepp > accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                for param_group in optimizer.param_groups:
                    print("Current learning rate:", param_group['lr'])
                writer.add_scalar('accumloss', loss.item() * accumulation_steps, stepp/accumulation_steps)
                scheduler.step()
                epoch_loss += loss.item()
                num_batches += 1
                avg_epoch_loss = epoch_loss / num_batches
        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                images,labels = batch["image"].to(device), batch["label"].to(device)
                output  = model(images)
                validation_loss = criterion(output, labels)
                validation_stepp += 1
                writer.add_scalar('validation_loss', validation_loss.item(), validation_stepp)
        if epoch % 5 == 0 and dist.get_rank() == 0: 
            save_checkpoint(model,optimizer,epoch,avg_epoch_loss,f"/data/vit/checkpoints/checkpoint_epoch_{epoch+1}.pt")
  
    cleanup()





if __name__ == "__main__":
    #ds = load_dataset("benjamin-paine/imagenet-1k-256x256")
    #train_ds = ds["train"]
    #test_ds = ds["test"].select(range(1000)) 
    #ds = load_dataset("benjamin-paine/imagenet-1k-256x256")
    #test_ds = ds["validation"]
    #ds_test = test_ds.map(
    #    transform_batch,
    #    batched=True,
    #    batch_size=2,
    #    num_proc=128,
    #)
    #ds_train = train_ds.map(
    #    transform_batch,
    #    batched=True,
    #    batch_size=2,
    #    num_proc=128,
    #)
    #ds_test = ds_test.shuffle(seed=42)
    #ds_test.save_to_disk("/mnt/mofs3/fs/josh/vit/imagenet_preprocessed_validation")
    #world_size = 16
    #mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    ray.init(address="auto") 
    trainer = TorchTrainer(
      train,
      scaling_config=ScalingConfig(num_workers=16, use_gpu=True),
    )
    #rank = int(os.environ["RANK"])
    #world_size = int(os.environ["WORLD_SIZE"])
    #train(rank, world_size)
    result = trainer.fit()
    print(result.metrics)
