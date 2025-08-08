import torch
from torch import nn

# 1) 사전(vocab)에서 단어 → 인덱스 가져오기
word = "great"  # 예시 단어
token_idx = vocab[word]  # vocab: {"great": 123, ...}

# 2) 모델의 embedding 레이어 정의 (임베딩 차원 D=64)
#    예: model.embedding = nn.Embedding(num_embeddings, 64)
embedding_layer: nn.Embedding = model.embedding

# 3) 임베딩 벡터 추출
with torch.no_grad():
    emb_tensor = embedding_layer(torch.tensor([token_idx]))  # shape: (1, 64)
    emb_vector = emb_tensor.squeeze(0).cpu().numpy()  # shape: (64,)

# 4) 64차원 값 출력
print("Embedding vector for word:", word)
print(emb_vector.tolist())  # 64개의 실수 값이 리스트 형태로 출력됩니다.
