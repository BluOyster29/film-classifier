from torch.utils.data import Dataset

class film_dataset(Dataset):
    
    def __init__(self, df, wrd2idx, genre2idx):
        
        self.film_id = []
        self.genre = []
        self.plot = []
        self.poster = []
        self.failed = []
        
        self.wrd2idx = wrd2idx
        self.genre2idx = genre2idx
        
        for col, row in tqdm(df.iterrows(), total=len(df)):
            self.film_id.append(row['id'])
            self.genre.append(encode_genre(row['genre'], genre2idx))
            self.plot.append(encode_plot(row['plot'], wrd2idx))
            self.poster.append(process_image(row['poster_path'], True))
            self.failed.append(row)
            
    
    def __getitem__(self, idx):
        
        return {
            'film_id' : self.film_id[idx],
            'plot'    : self.plot[idx],
            'poster'  : self.poster[idx],
            'genre'   : self.genre[idx]
        }
        
    def __len__(sef):
        return len(film_id)
    