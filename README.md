# Titanic — Logistic Regression (noldan, kutubxonasiz)

Ushbu loyihada logistic regression modeli hech qanday tayyor ML kutubxonasidan (masalan sklearn) foydalanmasdan, faqat numpy yordamida noldan yozildi. Maqsad — Titanic yo'lovchisi omon qolgan-qolmaganini (Survived: 0/1) bashorat qilish.

## Ma'lumotlar

Titanic dataset'idan 3 ta feature ishlatildi:
- Pclass — bilet klassi (1, 2, 3)
- Sex — jinsi
- Age — yoshi

Target: Survived (0 — omon qolmagan, 1 — omon qolgan)

## Preprocessing (ma'lumotni tayyorlash)

1. Age ustunidagi bo'sh (NaN) qiymatlar median yosh bilan to'ldirildi
2. Sex ustuni label encoding orqali songa aylantirildi (male=0, female=1)
3. Barcha feature'lar standartlashtirildi (mean=0, std=1):

$$x_{norm} = \frac{x - mean(x)}{std(x)}$$

4. Ma'lumot train (80%) va test (20%) qismlarga bo'lindi

## Model — qanday ishlaydi (kutubxonasiz)

### 1. Chiziqli qism

$$z = w_1 \cdot Pclass + w_2 \cdot Sex + w_3 \cdot Age + b$$

### 2. Sigmoid — z ni 0-1 oralig'idagi ehtimollikka aylantirish

$$p = \sigma(z) = \frac{1}{1+e^{-z}}$$

### 3. Loss — Binary Cross-Entropy

$$L = -\frac{1}{n}\sum_{i=1}^{n} \big[y_i \log(p_i) + (1-y_i)\log(1-p_i)\big]$$

### 4. Gradientlar (qo'lda chiqarilgan, chain rule orqali)

$$\frac{\partial L}{\partial w} = \frac{1}{n} X^T (p - y)$$

$$\frac{\partial L}{\partial b} = \frac{1}{n} \sum (p - y)$$

### 5. Gradient Descent — 1000 epoch davomida

$$w = w - \alpha \cdot \frac{\partial L}{\partial w}$$

$$b = b - \alpha \cdot \frac{\partial L}{\partial b}$$

bu yerda α (alpha) — learning rate (qadam kattaligi).

Har bir epochda model bashorat qiladi → xatoni o'lchaydi → gradientni hisoblaydi → w va b ni bir oz to'g'irlaydi. Shu jarayon 1000 marta takrorlanadi, natijada model asta-sekin to'g'ri qiymatlarga yaqinlashadi.

## Natijalar

| Ko'rsatkich | Qiymat |
|---|---|
| Train Loss (yakuniy) | 0.4562 |
| Test Accuracy | 78.77% |
| Test AUC | 0.859 |

## O'rganilgan qonuniyatlar

Model og'irliklari (w) shuni ko'rsatdi:
- Sex eng katta musbat ta'sirga ega — ayol bo'lish omon qolish ehtimolini sezilarli oshiradi ("ayollar va bolalar birinchi" qoidasi)
- Pclass manfiy ta'sirga ega — klass raqami oshgani sari (3-klassga yaqinlashgani sari) omon qolish ehtimoli kamayadi
- Age kichik manfiy ta'sirga ega — yosh oshgani sari omon qolish ehtimoli biroz kamayadi

## Fayllar

- titanic_logistic_regression.ipynb — to'liq kod (preprocessing, model, training, evaluation)

## Ishlatilgan kutubxonalar

Faqat numpy (matematik amallar) va pandas (ma'lumotni o'qish/tozalash uchun). Model logikasi (sigmoid, loss, gradient, gradient descent) to'liq qo'lda yozilgan, hech qanday sklearn yoki shunga o'xshash ML kutubxonasi ishlatilmagan.
