---
title: "第一篇：搭好了博客"
date: 2026-06-19T16:00:00+08:00
draft: false
math: true                      # 这一篇需要数学公式，所以打开
tags: ["元", "hugo"]
categories: ["随笔"]
summary: "用 Hugo + PaperMod 搭的博客，顺便测一下数学公式和代码高亮。"
---

终于把博客搭起来了。这篇用来测一下平时写研究笔记会用到的几样东西。

## 数学公式

行内公式：当损失为 $\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell(f_\theta(x_i), y_i)$ 时，梯度下降的一步是 $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$。

行间公式（注意力机制）：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

> 想在某篇文章里用公式，记得在 front matter 里加一行 `math: true`。

## 代码高亮

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / d_k ** 0.5
    attn = F.softmax(scores, dim=-1)
    return attn @ v
```

代码块右上角有一键复制按钮。

## 图片

把图片放进 `static/images/`，然后这样引用：`![描述](/images/xxx.png)`。

## 下一步

- 写新文章：`hugo new posts/我的文章.md`
- 本地预览：`hugo server -D`，浏览器开 http://localhost:1313
- 发布：`git add -A && git commit -m "..." && git push`，GitHub Actions 自动部署
