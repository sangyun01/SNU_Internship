import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import pwlf


# -----------------------------
# 1. ApproxLayerNorm 정의
# -----------------------------
class ApproxLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, G=16):
        super().__init__()
        self.eps = eps
        self.G = G
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        # ✅ 사전 계산된 .npz 파일에서 불러오기
        sqrt_npz = np.load("pwl_sqrt.npz")
        self.sqrt_breaks = torch.tensor(sqrt_npz["breaks"], dtype=torch.float32)
        self.sqrt_slopes = torch.tensor(sqrt_npz["slopes"], dtype=torch.float32)
        self.sqrt_intercepts = torch.tensor(sqrt_npz["intercepts"], dtype=torch.float32)

        recip_npz = np.load("pwl_recip.npz")
        self.recip_breaks = torch.tensor(recip_npz["breaks"], dtype=torch.float32)
        self.recip_slopes = torch.tensor(recip_npz["slopes"], dtype=torch.float32)
        self.recip_intercepts = torch.tensor(
            recip_npz["intercepts"], dtype=torch.float32
        )

    def pwl_approx(self, x, breaks, slopes, intercepts):
        out = torch.zeros_like(x)
        for i in range(len(slopes)):
            mask = (x >= breaks[i]) & (x < breaks[i + 1])
            out[mask] = slopes[i] * x[mask] + intercepts[i]
        out[x >= breaks[-1]] = slopes[-1] * x[x >= breaks[-1]] + intercepts[-1]
        return out

    def float_to_q8_8(self, x):
        return torch.round(x * 256).to(torch.int16)

    def q8_8_to_float(self, x_q):
        return x_q.to(torch.float32) / 256

    def pairwise_variance_q8_8(self, x_q):
        # x_q: int32 tensor of shape (B, D)
        B, D = x_q.shape
        splits = torch.chunk(x_q, self.G, dim=-1)
        n_list = [s.shape[-1] for s in splits]
        mu_list = [torch.sum(s, dim=-1, keepdim=True) // s.shape[-1] for s in splits]
        M_list = [
            torch.sum((s - mu).to(torch.int64) ** 2, dim=-1, keepdim=True)
            for s, mu in zip(splits, mu_list)
        ]

        while len(mu_list) > 1:
            next_mu, next_M, next_n = [], [], []
            for i in range(0, len(mu_list), 2):
                mu1, mu2 = mu_list[i], mu_list[i + 1]
                M1, M2 = M_list[i], M_list[i + 1]
                n1, n2 = n_list[i], n_list[i + 1]
                delta = mu1 - mu2
                delta_term = (delta.to(torch.int64) ** 2 * n1 * n2) // (n1 + n2)
                M12 = M1 + M2 + delta_term
                mu12 = (mu1 * n1 + mu2 * n2) // (n1 + n2)
                next_mu.append(mu12)
                next_M.append(M12)
                next_n.append(n1 + n2)
            mu_list, M_list, n_list = next_mu, next_M, next_n

        var_q16 = M_list[0] // D  # Q16.0
        var_float = self.q8_8_to_float(var_q16 // 256)  # -> Q8.8 → float
        return var_float  # shape: (B, 1)

    def forward(self, x):
        # Step 1: convert to Q8.8
        x_q = self.float_to_q8_8(x).to(torch.int32)
        mu_q = torch.sum(x_q, dim=-1, keepdim=True) // x_q.shape[-1]
        x_centered_q = x_q - mu_q

        # Step 2: pairwise Q8.8 variance
        var = self.pairwise_variance_q8_8(x_q)

        # Step 3: apply PWL sqrt & reciprocal
        sqrt_var = self.pwl_approx(
            var + self.eps, self.sqrt_breaks, self.sqrt_slopes, self.sqrt_intercepts
        )
        inv_sqrt = self.pwl_approx(
            sqrt_var, self.recip_breaks, self.recip_slopes, self.recip_intercepts
        )

        # Step 4: apply normalization (note: centered input은 float로 변환 필요)
        x_centered = self.q8_8_to_float(x_centered_q)
        x_norm = x_centered * inv_sqrt
        return x_norm * self.weight + self.bias


# -----------------------------
# 2. LayerNorm 교체
# -----------------------------
def replace_layernorm_with_approx(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            parent = model
            subnames = name.split(".")
            for sub in subnames[:-1]:
                parent = getattr(parent, sub)
            setattr(parent, subnames[-1], ApproxLayerNorm(module.normalized_shape[0]))


# -----------------------------
# 3. SST-2 로드 (샘플 10개만)
# -----------------------------
dataset = load_dataset("glue", "sst2")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(example):
    return tokenizer(example["sentence"], padding="max_length", truncation=True)


tokenized = dataset.map(tokenize_function, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 샘플 10개만 추출
small_train = tokenized["train"].select(range(10))
small_eval = tokenized["validation"].select(range(10))


# -----------------------------
# 4. 모델 불러오기 및 패치
# -----------------------------
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
replace_layernorm_with_approx(model)

# -----------------------------
# 5. 훈련 설정
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results-small",
    evaluation_strategy="epoch",
    save_strategy="no",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs-small",
    logging_steps=1,
)

metric = load_metric("glue", "sst2")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_eval,
    compute_metrics=compute_metrics,
)

trainer.train()
results = trainer.evaluate()

print("\n=== [SST-2] Validation Accuracy (Sample 10개) ===")
print(f"Accuracy: {results['eval_accuracy']*100:.2f}%")
