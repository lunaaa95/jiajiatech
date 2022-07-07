import datetime
import pickle
import copy
import ipdb
import torch
from torch.utils.data import DataLoader

from LunaKVM.model import KeyValueMemoryNet, MultiHeadAttentionKVMNet
from stock.config import get_args
from stock.dataset import split_data
from stock.epoch import cross_entropy_loss as loss_func
from stock.epoch import train_epoch, test_epoch, predict
from stock.evaluate import calc_acc, evaluate_by_class


def main():
    args = get_args()
    device = 'cuda:' + str(args.cuda)

    memory, train_dataset, validate_dataset = split_data(
        path_to_pickle=args.input,
        smooth_factor=args.smooth_factor,
        cut_num=10,
        independent_mem=args.independent_mem,
        mem_ratio=args.mem_ratio,
        val_ratio=args.val_ratio
    )
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    validate_dataloader = DataLoader(dataset=validate_dataset, batch_size=200)
    memory_data = (memory['keys'].to(device), memory['values'].to(device))

    hidden_size = args.hidden_size
    input_size = memory_data[0].shape[-1]
    value_size = memory_data[1].shape[-1]

    model = KeyValueMemoryNet(
        input_size=input_size,
        value_size=value_size,
        hidden_size=hidden_size,
        num_layers=1
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    accuracy_list = []
    max_test_acc = 0
    best_epoch = 0
    epoch = 0
    while True:
        epoch += 1
        print(f"-------------------------------\nEpoch {epoch}")
        train_acc = train_epoch(train_dataloader, memory_data, model, loss_func, optimizer, calc_acc, device)
        accuracy = test_epoch(validate_dataloader, memory_data, model, evaluate_by_class, device)
        if accuracy > max_test_acc and train_acc > 0.5:
            max_test_acc = max(accuracy, max_test_acc)
            best_epoch = epoch
            best_result = predict(validate_dataloader, memory_data, model, device)
            best_model = copy.deepcopy(model.state_dict())
        print(f'Val acc: {accuracy:>8f}')
        print('best epoch ', best_epoch + 1, f'is max_test_acc {max_test_acc:>8f}')
        if train_acc > 0.6:
            break

    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = args.output + '_' + now + '.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(best_result, f)
    torch.save(best_model, args.model)
    with open('tmp/stock_memory'+'_'+now+'.pkl', 'wb') as f:
        pickle.dump(memory, f)
    print(f'{now} finished')


if __name__ == '__main__':
    main()
