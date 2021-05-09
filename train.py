import os
import torch
import torch.nn.functional as F
import csv
import glob
import argparse

from datasets.dataloader import build_loader


parser = argparse.ArgumentParser(description="Graph Pooling")
parser.add_argument('--model', type=str, default="SAGNet", help='model name')
parser.add_argument('--seed', type=int, default=777, help='seed')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--dataset', type=str, default='IMDB-MULTI', help='DD/NCI1/NCI109/Mutagenicity/PROTEINS')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='path to save result')

args = parser.parse_args()


def train(args, model, train_loader, val_loader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    min_loss = 1e10
    max_acc = 0
    patience_cnt = 0
    best_epoch = 0
    val_loss_values = []
    for epoch in range(args.epochs):
        loss_train = 0.0
        loss_dis = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            output = model(data)
            cls_loss = F.nll_loss(output, data.y)
            dis_loss = model.compute_disentangle_loss()
            loss = cls_loss + 0.001 * dis_loss 
            loss.backward()
            optimizer.step()
            loss_train += cls_loss.item()
            loss_dis += dis_loss.item() 
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        acc_train = correct / len(train_loader.dataset)
        val_acc, val_loss = test(args, model, val_loader)
        if val_acc > max_acc:
            max_acc = val_acc
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train), 'loss_dis: {:.6f}'.format(loss_dis),
              'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(val_loss),
              'acc_val: {:.6f}'.format(val_acc), 'max_acc: {:.6f}'.format(max_acc))

        val_loss_values.append(val_loss)
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), 'save/' + args.dataset + '/' + str(args.seed) + '.pth')
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

    print('Optimization Finished!')


def test(args, model, loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(output, data.y, reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)


def main():
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        args.device = "cuda"
        print('cuda')
    else:
        args.device = "cpu"

    train_loader, val_loader, test_loader = build_loader(args)
    
    if args.model == "SAGNet":
        from models.SAGNet import SAGNet, Config
        config = Config()
        model = SAGNet(config, args)
    elif args.model == "GSANet":
        from models.GSANet import GSANet, Config
        config = Config()
        model = GSANet(config, args)
    elif args.model == "HGPSLNet":
        from models.HGPSLNet import HGPSLNet, Config
        config = Config()
        model = HGPSLNet(config, args)
    elif args.model == "ASAPNet":
        from models.ASAPNet import ASAPNet, Config
        config = Config()
        model = ASAPNet(config, args)
    model.to(args.device)

    train(args, model, train_loader, val_loader)

    model.load_state_dict(torch.load('save/' + args.dataset + '/' + str(args.seed) + '.pth'))
    test_acc, test_loss = test(args, model, test_loader)
    print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))
    
    with open('result.txt', 'a') as f:
        f.write(str(test_acc) + '\n')


if __name__ == '__main__':
    main()
