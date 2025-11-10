#!/usr/bin/env python3
import os, json, torch, random, re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

log_file = "logs.txt"
progress_file = "progress.json"
articles_file = "articles.json"

def log(msg): 
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f: f.write(msg + "\n")

site_desc = os.getenv("SITE_DESC", "a blog")
topics_input = os.getenv("TOPICS", "")
rewrite_count = int(os.getenv("REWRITE_COUNT", "0"))
generate_count = int(os.getenv("GENERATE_COUNT", "15"))
total = rewrite_count + generate_count

log(f"Target: {total} posts (rewrite {rewrite_count}, generate {generate_count})")

device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
model.eval()
similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

seen_title_hashes = set()
title_embeddings = []
articles = []

def generate_title(topic):
    prompt = f"6–12 word SEO title: '{topic}'. Site: {site_desc}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=50, temperature=0.9, do_sample=True)
    title = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return title if 6 <= len(title.split()) <= 12 else "Career Tips in Your Industry"

def generate_article(title):
    prompt = f"Write 420+ word post titled '{title}' for: {site_desc}. Include tips, story, CTA."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    out = model.generate(**inputs, max_new_tokens=1100, temperature=0.9, min_length=400, no_repeat_ngram_size=3)
    text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return text

# Generate titles
seeds = [t.strip() for t in topics_input.split("\n") if t.strip()] if topics_input else []
if not seeds:
    prompt = f"30 blog topics for: '{site_desc}'. 4–8 words. Numbered list."
    out = model.generate(tokenizer(prompt, return_tensors="pt").to(device), max_new_tokens=800)
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    seeds = [m.group(1) for m in re.finditer(r"\d+\.\s*(.+)", raw) if 4 <= len(m.group(1).split()) <= 8]

titles = []
for seed in seeds * 5:
    if len(titles) >= total: break
    t = generate_title(seed)
    h = hash(t.lower())
    if h in seen_title_hashes: continue
    emb = similarity_model.encode(t, convert_to_tensor=True)
    if any(util.cos_sim(emb, e).item() > 0.88 for e in title_embeddings): continue
    titles.append(t)
    seen_title_hashes.add(h)
    title_embeddings.append(emb)

# Generate articles
progress = {"total": total, "done": 0, "current": "", "percent": 0}
for i, title in enumerate(titles[:total], 1):
    progress.update({"current": title, "done": i-1, "percent": int((i-1)/total*100)})
    with open(progress_file, "w") as f: json.dump(progress, f)
    content = generate_article(title)
    articles.append({"title": title, "content": content})
    progress.update({"done": i, "percent": int(i/total*100)})
    with open(progress_file, "w") as f: json.dump(progress, f)

with open(articles_file, "w", encoding="utf-8") as f: json.dump(articles, f, indent=2)
progress.update({"percent": 100, "current": "Complete"})
with open(progress_file, "w") as f: json.dump(progress, f)
log("SUCCESS")
