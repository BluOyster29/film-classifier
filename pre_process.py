import pandas as pd, torch, re
from tqdm.notebook import tqdm
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision import transforms
from film_dataset import film_dataset



invTrans = transforms.Compose([
                                transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                                     std=[1/0.229, 1/0.224, 1/0.225]),
                               ])

def get_genres(genre):
    genre_set = []

    for i in genre:

        genres = i.split(',')

        for g in genres:

            g = g.strip()

            if g not in genre_set:
                genre_set.append(g)

        idx2genre = dict(enumerate(genre_set))
        genre2idx = {g : idx for idx, g in idx2genre.items()}

    return idx2genre, genre2idx

def count_genre(genre, genre2idx):

    genre_counts = {genre : 0 for genre in genre2idx.keys()}

    for i in genre:

        genres = i.split(',')

        for g in genres:

            g = g.strip()

            genre_counts[g] += 1

    return genre_counts

def encode_genre(genre, genre2idx):

    genre = genre.split(',')
    encoded_genre = torch.LongTensor([genre2idx[g.strip()] for g in genre])

    return encoded_genre

def reg_remove(plot):
    remove_non_words = re.compile(r'[^\w -]')
    clean = re.sub(remove_non_words, '', plot)
    return clean

def build_vocab(plots):

    vocab = {}
    processed_plots = []

    for plot in tqdm(plots):

        plot = reg_remove(plot.lower()).split(' ')
        plot.insert(0, '<start>')
        plot.append('<end>')

        for token in plot:

            if token not in vocab:
                vocab[token] = len(vocab) +1

        processed_plots.append(plot)

    idx2wrd = {idx : wrd for wrd,idx in vocab.items()}

    return vocab, idx2wrd, processed_plots

def encode_plot(plot, wrd2idx):

    encoded_plot = []

    for token in plot:

        if token in wrd2idx:
            encoded_plot.append(wrd2idx[token])

        else:
            encoded_plot.append(len(wrd2idx)+1)

    return encoded_plot

def process_image(filename, from_path):
    
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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

def output_image(image, filename):

    image = ToPILImage()(invTrans(image))
    image.save(filename)


if __name__ == '__main__':
    main()
