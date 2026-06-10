---
library_name: peft
model_name: llama1b
tags:
- base_model:adapter:C:\Users\merve\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct-bnb-4bit\snapshots\fcecd050a91025151bbf518e41a5577c356efdce
- lora
- sft
- transformers
- trl
- unsloth
licence: license
base_model: C:\Users\merve\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct-bnb-4bit\snapshots\fcecd050a91025151bbf518e41a5577c356efdce
pipeline_tag: text-generation
---

# Model Card for llama1b

This model is a fine-tuned version of [None](https://huggingface.co/None).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- PEFT 0.19.1
- TRL: 0.24.0
- Transformers: 5.5.0
- Pytorch: 2.12.0.dev20260408+cu128
- Datasets: 4.3.0
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```