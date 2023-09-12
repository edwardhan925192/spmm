import argparse
from tqdm import tqdm
import torch
import numpy as np
import time
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
import datetime
from spmm_custom_dataset import SMILESDataset_SHIN_MLM, SMILESDataset_SHIN_HLM, FEATUREDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import torch.nn as nn
from xbert import BertConfig, BertForMaskedLM
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class SPMM_regressor(nn.Module):
    def __init__(self,  tokenizer=None, config=None):
        super().__init__()
        self.tokenizer = tokenizer

        bert_config = BertConfig.from_json_file(config['bert_config_text'])
        self.text_encoder = BertForMaskedLM(config=bert_config)
        for i in range(bert_config.fusion_layer, bert_config.num_hidden_layers):  self.text_encoder.bert.encoder.layer[i] = nn.Identity()
        self.text_encoder.cls = nn.Identity()
        text_width = self.text_encoder.config.hidden_size

        # Freeze BERT layers
        for param in self.text_encoder.parameters():
            param.requires_grad = True

        # Projecting features denoising with additional dropout for regularization purpose
        self.feature_proj = nn.Sequential(
            nn.Linear(config['feature_dim'], config['feature_proj_dim']),
            nn.GELU(),
            nn.Dropout(config['dropout_prob']),  
        )

        self.reg_head = nn.Sequential(
            nn.Linear(text_width + config['feature_proj_dim'] , (text_width * 2) + config['feature_proj_dim'] ),
            nn.GELU(),
            nn.Linear((text_width * 2) + config['feature_proj_dim'], 1)
        )

        # Unfreeze feature_proj and reg_head
        for param in self.feature_proj.parameters():
            param.requires_grad = True
        for param in self.reg_head.parameters():
            param.requires_grad = True

        # ===================== param optim ===================== #
    def get_optimizer_params(self, base_lr,config):
    
      # Separate parameters of BERT and regression head
      bert_params = list(self.text_encoder.parameters())
      reg_head_params = list(self.feature_proj.parameters()) + list(self.reg_head.parameters())
      
      # Assign learning rates
      optimizer_grouped_parameters = [
          {"params": bert_params, "lr": base_lr * config['bert_lprob']},  # Use 0.1x the base learning rate for BERT
          {"params": reg_head_params, "lr": base_lr},    # Use the base learning rate for regression head
      ]
      return optimizer_grouped_parameters

    def forward(self, text_input_ids, text_attention_mask,features, value, eval=False):
        vl_embeddings = self.text_encoder.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state
        vl_embeddings = vl_embeddings[:, 0, :]
        
        # Project the features
        projected_features = self.feature_proj(features)
        
        # Concatenate with vl_embeddings
        combined_representation = torch.cat([vl_embeddings, projected_features], dim=-1)
        
        # Regression head
        pred = self.reg_head(combined_representation).squeeze(-1)

        if eval:    return pred
        lossfn = nn.MSELoss()
        loss = lossfn(pred, value)
        return loss

def train(model, data_loader, feature_loader, optimizer, tokenizer, epoch,  device ):
    # train
    model.train()

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 20
    step_size = 100    

    # Use zip to combine data_loader and feature_loader
    combined_loader = zip(data_loader, feature_loader)

    tqdm_data_loader = tqdm(combined_loader, total=len(data_loader), miniters=print_freq, desc=header)
    for i, ((text, value), features) in enumerate(tqdm_data_loader):
        optimizer.zero_grad()
        value = value.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)  # Move features to the device

        text_input = tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(device)

        # Pass features to the model
        loss = model(text_input.input_ids[:, 1:], text_input.attention_mask[:, 1:], features, value)
        loss.backward()
        optimizer.step()

        tqdm_data_loader.set_description(f'loss={loss.item():.4f}, lr={optimizer.param_groups[0]["lr"]:.6f}')        

@torch.no_grad()
def evaluate(model, data_loader, feature_loader, tokenizer, device, denormalize=None, is_validation=True):
    model.eval()
    preds = []

    # If it's validation, also collect ground truth values
    if is_validation:
        answers = []

    # Use zip to combine data_loader and feature_loader
    combined_loader = zip(data_loader, feature_loader)

    for (item, features) in combined_loader:
        # Unpack depending on the data loader's output for validation or test
        if is_validation:
            text, value = item
        else:
            text = item[0]
            value = None

        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)
        features = features.to(device, non_blocking=True)  # Move features to the device

        # Pass features to the model
        prediction = model(text_input.input_ids[:, 1:], text_input.attention_mask[:, 1:], features, value if value is not None else None, eval=True)
        preds.append(prediction.cpu())

        if is_validation:
            value = value.to(device, non_blocking=True)
            answers.append(value.cpu())

    preds = torch.cat(preds, dim=0)

    # If it's validation, compute RMSE
    if is_validation:
        answers = torch.cat(answers, dim=0)
        lossfn = nn.MSELoss()
        rmse = torch.sqrt(lossfn(preds, answers)).item()
        return rmse, preds, answers
    else:
        return preds

# ======================= Main body function ========================== #
def main(args, config):
    device = torch.device(args.device)
    print('DATASET:', args.name)

    # fix the seed for reproducibility
    seed = args.seed if args.seed else random.randint(0, 100)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    num_folds = 5
    all_val_rmses = []  # Store RMSE for each fold's validation set
    
    for fold in range(num_folds):

        print(f'======= FOLD {fold + 1} =======')

        # === Dataset === #
        print("Creating dataset for fold ", fold + 1)
        name = args.name    
        if name == 'HLM':
            dataset_train = SMILESDataset_SHIN_HLM('data/train.csv', mode='train', fold_num=fold)
            dataset_val = SMILESDataset_SHIN_HLM('data/train.csv', mode='val', fold_num=fold)
            dataset_test = SMILESDataset_SHIN_HLM('data/test.csv', mode='test')
            dataset_feature_train = FEATUREDataset('data/train_feature.csv', mode='train', fold_num=fold)
            dataset_feature_val = FEATUREDataset('data/train_feature.csv', mode='val', fold_num=fold)
            dataset_feature_test = FEATUREDataset('data/test_feature.csv', mode='test')

        elif name == 'MLM':
            dataset_train = SMILESDataset_SHIN_MLM('data/train.csv', mode='train', fold_num=fold)
            dataset_val = SMILESDataset_SHIN_MLM('data/train.csv', mode='val', fold_num=fold)
            dataset_test = SMILESDataset_SHIN_MLM('data/test.csv', mode='test')
            dataset_feature_train = FEATUREDataset('data/train_feature.csv', mode='train', fold_num=fold)
            dataset_feature_val = FEATUREDataset('data/train_feature.csv', mode='val', fold_num=fold)
            dataset_feature_test = FEATUREDataset('data/test_feature.csv',mode='test')

        train_loader = DataLoader(dataset_train, batch_size=config['batch_size_train'], num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset_val, batch_size=config['batch_size_test'], num_workers=2, pin_memory=True, drop_last=False)
        test_loader = DataLoader(dataset_test, batch_size=config['batch_size_test'], num_workers=2, pin_memory=True, drop_last=False)
        feature_train_loader = DataLoader(dataset_feature_train, batch_size=config['batch_size_train'], num_workers=2, pin_memory=True, drop_last=True)
        feature_val_loader = DataLoader(dataset_feature_val, batch_size=config['batch_size_test'], num_workers=2, pin_memory=True, drop_last=False)
        feature_test_loader = DataLoader(dataset_feature_test, batch_size=config['batch_size_test'], num_workers=2, pin_memory=True, drop_last=False)

        tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
        tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

        # === Model === #
        model = SPMM_regressor(config=config, tokenizer=tokenizer)
        model = model.to(device)

        # ============ Optimizer ============= #
        optimizer_grouped_parameters = model.get_optimizer_params(base_lr=args.lr, config=config)
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=config['optimizer']['weight_decay'])                

        scheduler_config = config['schedular']
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=scheduler_config['T_0'], T_mult=scheduler_config['T_mult'])

        max_epoch = config['schedular']['epochs']        
        best_valid = float('inf')

        # ============ Training start for this fold =============== #
        for epoch in range(0, max_epoch):

            print('TRAIN', epoch)
            train(model, train_loader, feature_train_loader, optimizer, tokenizer, epoch,  device)

            # ================== Validation datasets loaded ================== #
            val_rmse, _, _ = evaluate(model, val_loader, feature_val_loader, tokenizer, device, is_validation=True)
            print(f'VALID MSE for fold {fold + 1}, epoch {epoch}: %.4f' % val_rmse)
            if val_rmse < best_valid:
                best_valid = val_rmse

            # ================= scheduler step ================= # 
            scheduler.step()

        all_val_rmses.append(best_valid)

        # After all epochs for this fold, you can run the test if needed
        print(f'===== Testing for FOLD {fold + 1} =====')
        # test_preds = evaluate(model, test_loader, feature_test_loader, tokenizer, device, is_validation=False)

    # Calculate the average RMSE over the 5 validation sets
    average_rmse = sum(all_val_rmses) / num_folds
    print(f"Average validation RMSE across {num_folds} folds: {average_rmse:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_SPMM_20m.ckpt')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300.txt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=39, type=int)
    parser.add_argument('--name', default='esol', type=str)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--min_lr', default=5e-6, type=float)
    parser.add_argument('--epoch', default=25, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    args = parser.parse_args()

    cls_config = {
        'batch_size_train': args.batch_size,
        'batch_size_test': 16,
        'embed_dim': 256,
        'feature_dim': 197,
        'feature_proj_dim': 99,
        'bert_lprob':0.05,
        'dropout_prob': 0.2,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
        'schedular': {'lr': args.lr, 'epochs': args.epoch, 'min_lr': args.min_lr, 'T_0':2, 'T_mult':2},
        'optimizer': {'opt': 'adamW', 'lr': args.lr, 'weight_decay': 0.02}
    }
    main(args, cls_config)
