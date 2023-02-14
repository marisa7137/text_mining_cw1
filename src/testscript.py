## 纯粹是为了测试的脚本 最后会被删掉

from text_parser import TextParser

if __name__ == '__main__':
    t = TextParser()
    tensor = t.random_initialise_embedding(dim=15)
    print(len(t.raw_sentences))
    print(t.vocab.index('serfdom'))