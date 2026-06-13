import pickle
try:
    d = pickle.load(open('data/embed.pkl', 'rb'))
    dim = len(next(iter(d.values())))
    print('Embed dim:', dim)
except Exception as e:
    print('Error:', e)
