## 纯粹是为了测试的脚本 最后会被删掉
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from text_parser import TextParser
from model import Model
import torch.optim as optim
if __name__ == '__main__':
    lr = 0.1
    epochs = 10
    batch_size = 64
    t = TextParser()
    wi = t.get_word_indices(dim=10)
    random_perm = np.random.permutation(len(wi))
    test_size = int(0.1 * len(wi))
    train_indices = random_perm[test_size:]
    test_indices = random_perm[:test_size]
    train_data = [wi[i] for i in train_indices]
    test_data = [wi[i] for i in test_indices]
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # test_loader = DataLoader(test_data, batch_size=64, shuffle=True) 
    test_loader = DataLoader(test_data, shuffle=True) # batch size required?

    m = Model(pre_train_weight=None, vocab_size=len(t.vocab), embedding_dim=20, from_pre_train=False, freeze=False, bow=False, hidden_dim_bilstm=20, hidden_layer_size=30, num_of_classes=len(t.labels))
    optimizer = optim.Adam(m.parameters(), lr=lr)
    loss_function = torch.nn.NLLLoss()
    losses, train_accs = [], []
    m.train()
    for epoch in range(epochs):
        for train_labels, train_features in iter(train_loader):
            train_labels = train_labels.type(torch.LongTensor)
            if len(train_labels) != batch_size:
                continue
            output = m(train_features)
            loss = loss_function(output, train_labels)  # compute los
            optimizer.zero_grad()  # clean gradients
            loss.backward()  # backward pass
            optimizer.step()  # update weights
            losses.append(float(loss) / batch_size)  # average loss of the batch
        
        print("Train", f'epoch: {epoch}, loss: {loss.item()}, lr: {lr}')

    np.savetxt("loss_biLSTM.txt", losses)
    np.savetxt("acc_biLSTM.txt", train_accs)

    # plot_history(accList, lossList "./")

    # save the model
    torch.save(m, './biLSTM.pth')



# test once or validation?