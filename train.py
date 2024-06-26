import torch

from config import *
from utils import *
from model import *
# import matplotlib.pyplot as plt
if __name__ == '__main__':
    id2label, _ = get_label()

    train_dataset = Dataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=100, shuffle=True)

    dev_dataset = Dataset('dev')
    dev_loader = data.DataLoader(dev_dataset, batch_size=100)

    model = TextCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    dev_acc_list = []
    dev_acc = 0
    for e in range(10):
        for b, (input, mask, target) in enumerate(train_loader):
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            pred = model(input, mask)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每50个batch做一次
            # if b % 50 != 0:
            #     continue
            # torch.argmax dim = 1 返回每一行中最大值的索引
            y_pred = torch.argmax(pred, dim=1)
            # pred, true, target_names=None, output_dict=False
            report =evaluate(y_pred.cpu().data.numpy(), target.cpu().data.numpy(), output_dict=True)

            with torch.no_grad():
                dev_input, dev_mask, dev_target = iter(dev_loader).__next__()
                dev_input = dev_input.to(DEVICE)
                dev_mask = dev_mask.to(DEVICE)
                dev_target = dev_target.to(DEVICE)
                dev_pred = model(dev_input, dev_mask)
                dev_pred_ = torch.argmax(dev_pred, dim=1)
                dev_report = evaluate(dev_pred_.cpu().data.numpy(), dev_target.cpu().data.numpy(), output_dict=True)
            print(
                '>> epoch:', e,
                'batch:', b,
                'loss:', round(loss.item(), 5),
                'train_acc:', report['accuracy'],
                'dev_acc:', dev_report['accuracy'],
                # type(dev_report['accuracy'])
                # type(dev_report['accuracy'])
            )
            if dev_acc < dev_report['accuracy'] and dev_report['accuracy']>0.95:
                print("-----saving model------")
                # torch.save(model, MODEL_DIR + f'{e}.pth')
                torch.save(model, MODEL_DIR + 'the_best.pth')
                print("-----saving model end------")
                dev_acc = dev_report['accuracy']



