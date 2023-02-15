## 纯粹是为了测试的脚本 最后会被删掉

from sentence_rep import Sentence_Rep
from text_parser import TextParser
from word_embedding import Word_Embedding
if __name__ == '__main__':
    t = TextParser()
    wi = t.get_word_indices()
    print(t.vocab.index('serfdom'))
    wb = Word_Embedding(pre_train_weight=None, vocab_size=len(t.vocab), embedding_dim=30, freeze=True, from_pre_train=False)
    b1 = Sentence_Rep(word_embedding=wb, bow=True, embedding_dim=30, hidden_dim_bilstm=10)
    b2 = Sentence_Rep(word_embedding=wb, bow=False, embedding_dim=30, hidden_dim_bilstm=10)
    y1 = b1.forward(wi[0][1])
    y2 = b2.forward(wi[0][1])
    print(y1)
    print(y2)
