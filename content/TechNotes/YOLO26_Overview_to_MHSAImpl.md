---
title: YOLO26 Architectural Internals
draft: false
tags:
  - AI
  - ML
  - Deep-learning
  - Computer Vision
  - YOLO
---

For a technical and math-heavy deep-dive, refer to [YOLO26 Deep Dive](https://github.com/droyed/ml-tech-notes/blob/main/computer-vision/YOLO26_Deep_Dive.md).

# YOLO26 Internals: From overview to Multi-Head Self-Attention, Simply Explained
## I. YOLO26 — Architectural Overview

YOLO26 works in three stages, like an assembly line:

### Backbone

**The Backbone — "What's in this image?"** This is where the AI first "looks" at the raw photo. It scans for simple things first (edges, colors), then builds up to complex things (wheels, faces). YOLO26 uses a clever design called **C3k2 blocks** that splits the work into two paths — one does the heavy analysis, the other takes a shortcut — then merges them. This keeps the model lean and fast (up to 43% faster than older versions on regular CPUs). There's also a module called **SPPF** that helps the AI zoom out and understand the whole scene, not just tiny details.

### Neck (C2PSA)

**The Neck — "Let's make sense of things at every size"** Objects in photos can be tiny (a far-away bird) or huge (a truck up close). The Neck takes feature information from different stages and blends them together, so the AI is good at spotting both. Think of it like overlaying a detailed satellite photo with a labeled city map — you get sharp detail _and_ meaningful context at the same time. A special component called **C2PSA** (detailed in Section II) sharpens the network's focus at each scale.

### Head

**The Head — "Here are my final answers"** This is where the AI outputs its predictions: boxes drawn around objects, labels for what each object is, and how confident it is. YOLO26 uses **three separate heads**, each tuned for a different object size — small, medium, and large — so nothing gets missed regardless of scale.

---

## II. C2PSA Explained Simply

Imagine you're scanning a photo to find a dog in a busy park. Your eyes don't carefully examine every blade of grass — they quickly skim most of the scene and zoom in on the interesting parts. C2PSA teaches a neural network to do the same thing.

**C2PSA** stands for _Cross-Stage Partial Position-Sensitive Attention_, and it's a building block used in YOLO26 (a popular object detection model). Its whole job is to help the network figure out _where_ to focus in an image, without making the model slow.

### The Two Big Ideas Combined

C2PSA is really two concepts glued together: **C2** and **PSA**.

#### PSA

**PSA (Position-Sensitive Attention)** is the core idea. It's a mechanism that looks at every location in a feature map (think of this as a grid of compressed image information) and assigns each spot a score — "how important is this location?" Spots near object edges or meaningful structures get high scores; blurry background spots get low scores. This is the _what to focus on_ engine.

#### C2

**C2 (Cross-Stage Partial)** is the _efficiency wrapper_ around PSA. Rather than running the entire feature map through the expensive PSA process, C2 splits the data into two streams first:

- **One stream bypasses PSA entirely** — it's a shortcut that keeps the original data intact and helps gradients flow during training.
- **The other stream goes through PSA** to get the attention treatment.

So in short: **PSA is the attention brain, and C2 is the smart plumbing that makes it affordable to run.**

### The Full Flow, Simply

1. **Split** — incoming data is divided in two.
2. **Bypass** — one half passes through unchanged, preserving the original signal.
3. **Attend** — the other half gets scored by PSA (_"what matters here?"_).
4. **Refine** — the attention output is blended back with its own raw input and passed through small neural layers to deepen the learning.
5. **Reunite** — both halves are merged back together into one richer feature map.

The result is a feature map that knows _where_ the important stuff is, computed at roughly half the cost of running full attention — which is exactly why this design shows up in a speed-focused model like YOLO26.

![[YOLO26_1.png]]

Figure1: YOLO26 architecture at higher-level with the code flow - **Backbone** (C3k2 dual-path blocks → SPPF) → **Neck** (Feature Fusion → C2PSA with its Split/Bypass/Attend/Refine/Reunite inner flow) → **Head** (3 scale-specific detection heads) → final predictions.

---

## III. Position-Sensitive Attention (PSA) — Explained Simply

### The Problem: Convolutions Have Tunnel Vision

Imagine you're trying to identify a car in a photo, but you can only look through a tiny 3×3 pixel peephole at a time. That's essentially what a standard convolutional filter does — it's fast, but it can only see what's _right in front of it_. It has no idea what's happening on the other side of the image.

PSA was designed to fix this. Think of it as giving the network a **bird's-eye view** — it can see the _entire_ image at once and reason about how distant parts relate to each other.

### Why "Position-Sensitive"?

Most attention mechanisms in deep learning ask: _"What am I looking at?"_ (e.g., is this a wheel? an eye? a door?). They care about **what** features are present but throw away **where** they are spatially.

PSA asks a richer question: _**"What am I looking at, and exactly where is it?"**_

This matters enormously in object detection — knowing a wheel exists is useful, but knowing _where_ it sits relative to a chassis is what confirms "that's a car."

### How PSA Actually Works (Step by Step)

#### Step 1 — Flatten the image into a sequence

The feature map (a 3D grid of numbers: channels × height × width) gets unrolled into a flat list of "tokens" — one token per spatial location. Think of turning a chessboard into a single numbered list of all 64 squares. Each square keeps its information; we just reorganized it so attention can work on it.

#### Step 2 — Multi-Head Self-Attention (MHSA) ← the core engine

This is where PSA leverages Multi-Head Self-Attention (MHSA) — detailed in Section IV. In short, every spatial location compares itself against every other location to discover which parts of the image are most relevant to each other, and multiple "heads" do this in parallel, each learning different types of relationships.

> **In short: MHSA is the engine. PSA is the car.** PSA wraps MHSA in a vision-specific pipeline (flattening, positional encoding, residual shortcuts) to make it practical for images.

#### Step 3 — Feed-Forward Network (FFN)

After attention figures out _relationships_ between locations, a small two-layer network deepens the _meaning_ of each individual token using those newly discovered relationships. Think of it as: attention finds the connections, FFN thinks harder about what those connections mean.

#### Step 4 — Residual Shortcut

The output gets added back to the original input (a "skip connection"). This ensures fine-grained details like textures and edges — captured earlier in the network — don't get washed out as data flows through many layers.

### The Payoff: Smarter Detections

Without PSA, a model might see a circular texture, match it to "wheel," and confidently shout "Car!" — even if there's no chassis, no other wheels, nothing else car-like nearby. That's a **false positive**.

With PSA, the model can instantly cross-check: _"Are the other wheels at the right relative positions? Is there a chassis below?"_ If the spatial context doesn't add up, it suppresses the wrong guess. PSA essentially gives the model a **built-in sanity check** using the full layout of the scene.

### One-Line Summary

PSA plugs the Transformer's MHSA mechanism into a vision pipeline so that, instead of looking at one tiny patch at a time, the network can reason about how _every part of an image relates to every other part_ — spatially and simultaneously.

---

## IV. Multi-Head Self-Attention (MHSA)

Imagine you're reading the sentence: _"The cat sat on the mat because it was tired."_ To understand what "it" refers to, you need to connect it back to "cat." This is exactly what **self-attention** does — it helps a model figure out which words (or pixels, or tokens) are relevant to each other.

### First: Regular (Single-Head) Attention

Think of every token in your input as a person at a networking event. Each person has three things:

- A **Query** — "I'm looking for someone who knows about X"
- A **Key** — "Here's what I know about"
- A **Value** — "Here's the actual knowledge I can share"

The attention mechanism works by matching Queries to Keys to figure out _who should pay attention to whom_, then passing along the Values from the most relevant matches. The math keeps scores from blowing up (by dividing by √d), then converts them into nice probabilities (via softmax), and finally blends the Values accordingly.

### Now: Why "Multi-Head"?

A single attention head can only look for **one type of relationship at a time**. But real data has many simultaneous relationships. In an image of a dog, two pixels might be related because they're the same color, _or_ because they're part of the same ear, _or_ because they're spatially close.

**Multi-Head Attention runs several attention heads in parallel**, where each head learns to focus on a _different type_ of relationship. Think of it like a team of specialists all examining the same image at once — one looking at color, another at shapes, another at objects. Each specialist works in their own smaller "subspace" of the data.

### Putting It All Together

Once all the heads are done, their outputs are simply **glued together (concatenated)** and then passed through one final learned layer (W⁰) that acts like a **committee chair** — it blends all the specialists' opinions into one unified, rich representation.

The best part? Running multiple heads doesn't cost much more than running one, because each head works on a _smaller slice_ of the data. You get multiple perspectives essentially for free.

### Why It's Powerful

|Single-Head Attention|Multi-Head Attention|
|---|---|
|One perspective at a time|Many perspectives simultaneously|
|May miss subtle patterns|Different heads catch different patterns|
|Less expressive|Richer, more nuanced representations|

![[YOLO26_2.png]]

Figure2: Going deeper into PSA's internals: Flatten → **MHSA** (with the Q/K/V diagram, attention math, and parallel head visualization) → FFN → Residual shortcut — then loops back to C2PSA's reunite step. The MHSA node is highlighted with a gradient border glow as the destination endpoint of the pipeline trace.

**The one-line takeaway:** MHSA lets a model look at its input through many specialized "lenses" at once, each learning to notice different kinds of patterns, and then combines all those insights into one powerful understanding.

----

## V. MHSA - Ultralytics YOLO implementation deep-dive

In fast, real-world models like Ultralytics YOLO, the heavy math layers in MHSA are swapped out for lightweight **convolution operations** (tiny 1×1 and 3×3 filters). Same idea, just turbocharged for speed — important when you need to detect objects in video in real time.

Listed below is the attention module implementation in the Ultralytics library, which will be used for the explanation in the following section:

```python
class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
```

Source: [`block.py#L1271`](https://github.com/ultralytics/ultralytics/blob/v8.4.21/ultralytics/nn/modules/block.py#L1271) from [Ultralytics](https://github.com/ultralytics/ultralytics), licensed under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

### Explanation

It is helpful to note here that **self-attention acts as a smart "highlighter"** that allows the model to look at the "big picture" and figure out how every pixel in an image relates to every other pixel.

Here is a step-by-step breakdown of how this code physically implements that highlighting process inside a neural network.

#### Part 1: The Setup (The `__init__` function)

Before the AI can process an image, it needs to set up its tools.

- **`self.num_heads = num_heads` (Multi-Head Setup):** Instead of looking at the image with just one perspective, the model splits its focus into multiple "heads" (default is 8). Imagine having a team of 8 investigators looking at the same image—one looks specifically for colors, another looks for edges, and another looks for textures.
- **`self.qkv = Conv(...)` (The Information Generator):** This is a 1x1 convolutional layer. Its job is to take the input image data and simultaneously split it into three distinct pieces of information for every single pixel: **Queries (Q)**, **Keys (K)**, and **Values (V)**.
- **`self.scale = self.key_dim**-0.5` (The Stabilizer):** When multiplying large matrices of numbers, the values can explode and break the model's learning process. This scale acts as a mathematical stabilizer to keep the numbers small and manageable.
- **`self.pe = Conv(...)` (The Position Perceiver):** Standard self-attention is "position-blind," meaning it knows two pixels are related but forgets _where_ they actually are in the image. To fix this efficiently, modern optimized attention blocks use a depthwise convolution (acting as a "position perceiver") to implicitly encode spatial information back into the data without slowing the model down.

#### Part 2: The Action (The `forward` function)

This is where the actual math happens when an image (`x`) passes through the module.

**1. Flattening the Image**

```
B, C, H, W = x.shape
N = H * W
```

The model takes the 2D image (Height and Width) and flattens it into a 1D sequence of pixels (`N`). Self-attention needs to compare every pixel to every other pixel in a massive grid, so treating the image as a sequence makes the math possible.

**2. Generating Queries, Keys, and Values**

```
qkv = self.qkv(x)
q, k, v = qkv.view(...).split(...)
```

The model uses the generator it set up earlier to create the `q`, `k`, and `v` matrices.

- **Query (q):** This represents what a specific pixel is "looking for" (e.g., a pixel on a car door is looking for a window).
- **Key (k):** This represents what a pixel "contains" (e.g., "I am made of glass").
- **Value (v):** This is the actual underlying visual data of the pixel.

**3. Matchmaking (Calculating Attention Scores)**

```
attn = (q.transpose(-2, -1) @ k) * self.scale
```

This is the core of self-attention. The model uses a dot-product (`@`) to mathematically multiply the **Queries** against the **Keys**. If a Query perfectly matches a Key, the resulting number (the "attention score") is very high. This means the network has discovered that those two pixels are highly related, even if they are on opposite sides of the image. It then applies the `self.scale` to keep the math stable.

**4. The Highlighter (Softmax)**

```
attn = attn.softmax(dim=-1)
```

The `softmax` function converts the raw scores into percentages (probabilities) that add up to 100%.

- If two pixels are highly related, they might get a score of 0.95 (95% attention).
- If they are unrelated background noise, they get a score of 0.01 (1% attention). This creates an "attention map" which acts as the literal highlighter, telling the model exactly where to focus its mental energy.

**5. Applying the Highlight and Spatial Memory**

```
x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
```

Here, the model multiplies the original **Values (`v`)** by the **Attention Percentages (`attn`)**.

- Irrelevant pixels are multiplied by numbers close to 0, effectively erasing them (ignoring the background).
- Important pixels are multiplied by numbers close to 1, preserving and highlighting them. Then, it reshapes the sequence back into a 2D image (`view(B, C, H, W)`) and adds the `self.pe` (Position Perceiver) so the model remembers exactly where these highlighted features belong in the physical space of the image.

**6. Final Packaging**

```
x = self.proj(x)
return x
```

Finally, the newly highlighted, context-aware image map is passed through a final projection layer (`self.proj`). This smoothly packages the data back into the standard channel size so it can be passed on to the next layer of the neural network.

---

### Visual representation

Let's set up a flowchart to analyze the different components for a visual representation of the forward call.

**Tracing the tensor shapes:**

1. `x` → `(B, C, H, W)` — input image features
2. `qkv = self.qkv(x)` → `(B, h, H, W)` where `h = C + 2·key_dim·heads`
3. `.view(B, heads, key_dim*2 + head_dim, N)` — reshape, `N = H·W`
4. `.split(...)` → `q: (B, heads, key_dim, N)`, `k: (B, heads, key_dim, N)`, `v: (B, heads, head_dim, N)`
5. `q.T @ k` → `(B, heads, N, N)` — the full pixel-to-pixel attention matrix
6. `* scale` → still `(B, heads, N, N)`
7. `.softmax(dim=-1)` → `(B, heads, N, N)` — probabilities
8. `v @ attn.T` → `(B, heads, head_dim, N)` → `.view(B, C, H, W)`
9. `+ self.pe(v.reshape(B, C, H, W))` — add positional encoding
10. `self.proj(x)` → `(B, C, H, W)` — final output

A few design choices worth noting:

**The dual V path** is the most interesting architectural detail — Value splits into two branches: one feeds into the attention multiplication (weighted by the softmax scores), and the other gets reshaped back to 2D and run through the depthwise PE convolution. These two paths merge via addition before the final projection. This is how the module re-injects spatial awareness after the "position-blind" attention computation.

**Tensor shapes are traced at every node**, so you can follow exactly how `(B, C, H, W)` gets reshaped into sequences, split into Q/K/V with different dimensionalities (Key uses `key_dim`, Value uses `head_dim`), expanded into the `(B, h, N, N)` attention matrix, and then collapsed back to the original spatial shape.

![[YOLO26_attention_module_forward_pass.png]]

Figure3: Deep-dive of Ultralytics YOLO implementation of MHSA

## VI. The Full Picture — Zooming Back Out

We started at 30,000 feet and drilled all the way down to individual tensor reshapes inside a single `forward()` call. At the bottom of that rabbit hole sits MHSA — the real workhorse of the entire pipeline. It's the mechanism that lets the model ask _"how does every part of this image relate to every other part?"_ and answer through the elegant Q/K/V dance of dot products, softmax probabilities, and value-weighted blending. Everything above it in the architecture — C3k2's dual-path splits, C2PSA's bypass channel, the three scale-tuned detection heads — is essentially plumbing designed to feed MHSA the right data and carry its insights to the final predictions as cheaply as possible.

That's the real lesson of YOLO26's internals: MHSA gives the model a powerful, scene-wide understanding that convolutions alone can't match — but it's the _clever plumbing_ wrapping it that makes the whole thing fast enough to run on a video feed in real time.