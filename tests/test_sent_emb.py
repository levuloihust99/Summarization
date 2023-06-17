import torch
# from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# model = SentenceTransformer("NtDNlp/sentence-embedding-vietnamese")
model = AutoModel.from_pretrained("NtDNlp/sentence-embedding-vietnamese")

tokenizer = AutoTokenizer.from_pretrained("NtDNlp/sentence-embedding-vietnamese")
input_sentence = ["Cho mình hỏi tên của bạn là gì?", "Bạn ơi tên bạn là gì đấy?", "Bạn ơi tên bạn là gì đấy? Hà Nội mùa này có đẹp không?", "Thằng này tên gì"]
inputs = tokenizer(input_sentence, padding=True, return_tensors="pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

model.eval()

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    embeddings = mean_pooling(outputs, attention_mask)

embeddings = torch.nn.functional.normalize(embeddings, dim=1)
sim_score = torch.sum(embeddings[0] * embeddings[1])
pass
