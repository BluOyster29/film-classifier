from torch.utils.data import Dataset
import numpy as np, pandas as pd, re, torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def reg_remove(plot):
    remove_non_words = re.compile(r'[^\w -]')
    clean = re.sub(remove_non_words, '', plot)
    return clean

def process_image(filename, from_path):

    if from_path == False:

        input_image = Image.open(filename)
        transformed = transform(input_image)
        filename = filename.split('/')[-1][:-5]
        filename = 'data/processed_posters/{}-processed.jpeg'.format(filename)

        output_image(transformed, filename)
        return transformed

    else:

        transformed = transform(Image.open(filename))
        return transformed

class FilmDataset(Dataset):

    def __init__(self, df):

        self.df = pd.read_csv(df).dropna()

    def __getitem__(self, idx):

        return {
            'film_id' : '{}-{}'.format(self.df.loc[idx]['id'],self.df.loc[idx]['title']),
            'plot'    :         self.processed_plots[idx]
            'encoded_plot'    : self.encoded_plots[idx],
            'poster'  : process_image(self.df.loc[idx]['poster_path'], True),
            'genre'   : self.encode_genre(self.df.loc[idx]['genre'])
        }

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return "<Film Dataset\n {} Films with Posters>".format(len(self))

    def encode_plot(self, plot, wrd2idx):

        encoded_plot = []

        for token in plot:

            if token in wrd2idx:
                encoded_plot.append(wrd2idx[token])

            else:
                encoded_plot.append(wrd2idx['<unk>'])

        return encoded_plot

    def process_plots(self, build_vocab):

        plots = self.df['plot'].tolist()

        vocab = {'<pad>' : 0,
                 '<unk>' : 1,
                 '<start>' : 2,
                 '<end>' : 3}

        processed_plots = []

        for plot in tqdm(plots):

            plot = reg_remove(plot.lower()).split(' ')
            plot.insert(0, '<start>')
            plot.append('<end>')

            for token in plot:

                if token not in vocab:
                    vocab[token] = len(vocab)

            processed_plots.append(plot)

        idx2wrd = {idx : wrd for wrd,idx in vocab.items()}

        if build_vocab == True:
            self.wrd2idx = vocab
            self.idx2wrd = idx2wrd

        self.processed_plots = processed_plots

    def encode_plots(self):

        encoded = []

        for i in tqdm(self.processed_plots):
            encoded.append(torch.LongTensor(self.encode_plot(i, self.wrd2idx)))

        self.encoded_plots = pad_sequence(encoded, padding_value=0,batch_first=True)

    def set_encoders(self, wrd2idx, idx2wrd, genre2idx, idx2genre):

        self.wrd2idx = wrd2idx
        self.idx2wrd = idx2wrd
        self.genre2idx = genre2idx
        self.idx2genre = idx2genre

    def set_genres(self):
        genre = self.df['genre'].tolist()
        genre_set = []

        for i in genre:
            genres = i.split(',')
            for g in genres:
                g = g.strip()

                if g not in genre_set:
                    genre_set.append(g)

        self.idx2genre = dict(enumerate(genre_set))
        self.genre2idx = {g : idx for idx, g in self.idx2genre.items()}

    def encode_genre(self, genre):

        genre = genre.split(',')
        empty = np.zeros(len(self.genre2idx))
        for g in genre:
            empty[self.genre2idx[g.strip()]] = 1

        return torch.LongTensor(empty)

    def extract_encoders(self):
        return self.wrd2idx, self.idx2wrd, self.genre2idx, self.idx2genre

    def validate_dataset(self):
        self.failed=[]
        for i in tqdm(range(len(self))):

            try:
                self[i]

            except:
                self.failed.append(i)
                self.df = self.df.drop(i)

    def info(self):
        print('''
                1: dataset.process_plots(True)
                2: dataset.encode_plots()
                3: wrd2idx, idx2wrd, genre2idx, idx2genre = train_dataset.extract_encoders()
                4: (Optional) train_dataset.validate_dataset()
              ''')
