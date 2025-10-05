# **Fair Adaptive Scaling (FAS) – Experiment** 

## **About** 📌  
This project was a small experiment to understand Fair Adaptive Scaling (FAS) in practice and see how it helps reduce bias in classification tasks.  

I worked with 3,000 images:  
-  1,500 male  
-  1,500 female  

After preprocessing, I fine-tuned a ResNet18 model.  

## **Experiment Steps** 🔬  
1. Trained the baseline model on the dataset.  
2. Calculated accuracy scores for individual classes on 1,000 new test images.  
3. Identified the underperforming class (male).  
4. Retrained the model using Fair Adaptive Scaling (FAS) to give more balanced attention to groups.  

## **My Understanding of FAS** 🧠  
FAS does not treat every training sample equally. Instead, it adapts how much attention the model pays to each sample during training.  

- **Individual-level weighting**  
  - Tracks how difficult each sample has been over time (using an exponential moving average of its cross-entropy loss).  
  - Samples that remain hard to classify get a higher weight.  

- **Group-level weighting**  
  - Each group (e.g., male or female) has its own learnable parameter.  
  - If a group overall is underperforming, its parameter increases so the model assigns more weight to that group’s samples.  

In my implementation, I combined these two signals with a parameter **c**:  
- Half weight from the sample’s difficulty.  
- Half weight from the group parameter.  
- The weights were normalized and clipped to stay stable.

# 🧠 Mathematical Formulation and Implementation

This approach improves **model equity** without sacrificing overall performance.

Let:

- \( N \): total number of training samples  
- \( K \): number of demographic groups  
- \( a_i \): group ID for sample \( i \)  
- \( L_i \): cross-entropy loss for sample \( i \)  
- \( \ell_i^{(t)} \): exponential moving average (EMA) of past losses for sample \( i \)  
- \( \beta_a \): learnable group weight for group \( a \)  
- \( c \in [0,1] \): mixing coefficient between individual and group scaling  

---

### 1️⃣ Base Loss

Each sample has a cross-entropy loss:

\[
L_i = - \sum_{k} y_{i,k} \log(p_{i,k})
\]

---

### 2️⃣ Exponential Moving Average of Loss

To track sample difficulty over time:

\[
\ell_i^{(t)} = (1 - \alpha)\,\ell_i^{(t-1)} + \alpha\,L_i
\]

where \( \alpha \) is the smoothing factor (`ema_alpha`).

---

### 3️⃣ Individual-Level Scaling

Normalize recent sample losses and map them to a stable range using a sigmoid function:

\[
\text{indiv}_i = 0.5 + \sigma\!\left(\frac{\ell_i^{(t)} - \mu_{\ell}}{\sigma_{\ell} + \epsilon}\right)
\]

Samples with higher recent losses (harder to classify) receive higher weights.

---

### 4️⃣ Group-Level Scaling

Each demographic group learns its own adaptive weight parameter:

\[
\beta_a = \text{Embedding}(a)
\]

If a group overall performs poorly, the optimization increases its corresponding \( \beta_a \).

---

### 5️⃣ Combined Adaptive Weight

The total weight for each sample \( i \) is a linear combination of the two signals:

\[
w_i = c \cdot \text{indiv}_i + (1 - c) \cdot \beta_{a_i}
\]

- \( c \) controls the trade-off between individual fairness and group fairness.  
- The weights are normalized and clipped:

\[
w_i = \text{clip}\!\left(\frac{w_i}{\overline{w}},\, w_{\min},\, w_{\max}\right)
\]

---

### 6️⃣ Fair Adaptive Scaling Loss

Finally, the **FAS loss** is computed as:

\[
\mathcal{L}_{\text{FAS}} = \frac{1}{N} \sum_{i=1}^{N} w_i \, L_i
\]

When `enable_fas=False`, the loss reduces to standard cross-entropy.

---

## 🧮 Statistical Evaluation (Simplified)

In this project:
- Model performance was measured by **class-wise accuracy** (Male vs Female).  
- The **fairness gap** was defined as the absolute difference in accuracy between groups.  


## **Results** 📊  

| Model    | Male Accuracy | Female Accuracy | Gap   |
|----------|--------------:|----------------:|-------|
| Baseline | 73.74%        | 87.66%          | 13.92% |
| With FAS | 88.19%        | 87.75%          | 0.44%  |

✅ With FAS, the male accuracy improved significantly, while female accuracy remained stable.  
The performance gap between groups nearly disappeared.  
