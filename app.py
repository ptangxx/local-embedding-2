import torch
import os
import pickle
import datetime
import numpy as np
from sentence_transformers import SentenceTransformer

def log(message):
    now = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    print(f'[{now}] {message}')

def get_basename(path):
    return os.path.splitext(os.path.basename(path))[0]

def output_vector_file(path, sentence):
    vector = model.encode(sentence)
    with open(path, 'wb') as file:
        pickle.dump(vector, file)
    return vector

model_name = 'hotchpotch/static-embedding-japanese'
sentence_dir = './sentence/'
vector_dir = './vector/'

log('CUDAの有効化中...')
if torch.cuda.is_available():
    device = torch.device('cuda')
    log(f'CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')
    log('CUDA is not available. Using CPU.')

log('モデルの読み込み中...')
model = SentenceTransformer(model_name, device=device)

log('ベクターファイルの読み込み中...')
datas = {}
for file_name in os.listdir(vector_dir):
    path = vector_dir + file_name
    with open(path, 'rb') as file:
        vector = pickle.load(file)
    datas[get_basename(file_name)] = {
        'path': path,
        'updated_at': os.path.getmtime(path),
        'vector': vector
    }

log('センテンスファイルの読み込み中...')
for file_name in os.listdir(sentence_dir):
    path = sentence_dir + file_name
    basename = get_basename(file_name)
    updated_at = os.path.getmtime(path)
    with open(path, 'r', encoding='utf-8') as file:
        sentence = file.read()
    if basename not in datas:
        log(f'+{file_name}のベクターファイルを新規作成中...')
        vector_file_path = vector_dir + basename + '.pkl'
        vector = output_vector_file(vector_file_path, sentence)
        datas[basename] = {
            'path': vector_file_path,
            'updated_at': updated_at,
            'vector': vector
        }
    elif datas[basename]['updated_at'] < updated_at:
        log(f'+{file_name}のベクターファイルを更新中...')
        vector_file_path = vector_dir + basename + '.pkl'
        vector = output_vector_file(vector_file_path, sentence)
        datas[basename] = {
            'updated_at': updated_at,
            'vector': vector
        }
    datas[basename]['sentence'] = sentence

log('データ整理中...')
vectors_np = np.array([data['vector'] for data in datas.values()])
vectors_tensor = torch.tensor(vectors_np, device=device)
sentences = [data['sentence'] for data in datas.values()]

log('ベクトル類似度を計算中...')
query = '美味しいラーメン屋に行きたい'
query_vector_np = model.encode(query)
query_vector_tensor = torch.tensor(query_vector_np, device=device)
similarities = model.similarity(query_vector_tensor.unsqueeze(0), vectors_tensor)
# for i, similarity in enumerate(similarities[0].tolist()):
#     print(f'{similarity:.04f}: {sentences[i]}')
sorted_results = sorted(zip(similarities[0].tolist(), sentences), key=lambda x: x[0], reverse=True)
for similarity, sentence in sorted_results:
    print(f'{similarity:.04f}: {sentence}')

log('処理完了')
