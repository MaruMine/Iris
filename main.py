# coding: utf-8

import numpy as np

# ハイパーパラメータ
TRAIN_DATA_SIZE = 50  # 150個のデータのうちTRAIN_DATA_SIZE個を訓練データとして使用。残りは教師データとして使用。
HIDDEN_LAYER_SIZE = 6  # 中間層(隠れ層)のサイズ(今回は中間層は1層なのでスカラー)
LEARNING_RATE = 0.1  # 学習率
ITERS_NUM = 1000  # 繰り返し回数
DELTA = 0.01

# データを読み込み
# デフォルトで'#'の行をを飛ばすようになっている
x = np.loadtxt('iris.tsv', delimiter='\t', usecols=(0, 1, 2, 3))
raw_t = np.loadtxt('iris.tsv', dtype=int, delimiter='\t', usecols=(4,))
onehot_t = np.zeros([150, 3])
for i in range(150):
    onehot_t[i][raw_t[i]] = 1

train_x = x[:TRAIN_DATA_SIZE]
train_t = onehot_t[:TRAIN_DATA_SIZE]
test_x = x[TRAIN_DATA_SIZE:]
test_t = onehot_t[TRAIN_DATA_SIZE:]

# 重み・バイアス初期化
W1 = np.random.randn(4, HIDDEN_LAYER_SIZE) * np.sqrt(2 / 4)  # Heの初期値(ReLUのときはこれを使う)
W2 = np.random.randn(HIDDEN_LAYER_SIZE, 3) * np.sqrt(2 / HIDDEN_LAYER_SIZE)
b1 = np.zeros(HIDDEN_LAYER_SIZE)  # 初期値ゼロ　※ゼロから作るDeep Learningを見てこうしたので理由はわからない
b2 = np.zeros(3)

# ReLU関数
def relu(x):
    return np.maximum(x, 0)

# Softmax関数　※この関数だけネットを見たのでどう実装しているかわからない
def softmax(x):
    e = np.exp(x - np.max(x))
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    elif e.ndim == 2:
        return e / np.array([np.sum(e, axis=1)]).T
    else:
        raise ValueError

# 交差エントロピー誤差
def cross_entropy_error(y, t):
    if y.shape != t.shape:
        raise ValueError
    if y.ndim == 1:
        return - (t * np.log(y)).sum()
    elif y.ndim == 2:
        return - (t * np.log(y)).sum() / y.shape[0]
    else:
        raise ValueError

# 順伝搬
def forward(x):
    global W1, W2, b1, b2
    return softmax(np.dot(relu(np.dot(x, W1) + b1), W2) + b2)

# テストデータの結果
test_y = forward(test_x)
print((test_y.argmax(axis=1) == test_t.argmax(axis=1)).sum(), '/', 150 - TRAIN_DATA_SIZE)

# 学習ループ
for i in range(ITERS_NUM):
    # 順伝搬withデータ保存
    y1 = np.dot(train_x, W1) + b1
    y2 = relu(y1)
    train_y = softmax(np.dot(y2, W2) + b2)

    # 損失関数計算
    L = cross_entropy_error(train_y, train_t)

    if i % 100 == 0:
        print(L)

    # 勾配計算
    # 計算グラフで求めた式を使用
    a1 = (train_y - train_t) / TRAIN_DATA_SIZE
    b2_gradient = a1.sum(axis=0)
    W2_gradient = np.dot(y2.T, a1)
    a2 = np.dot(a1, W2.T)
    a2[y1 <= 0.0] = 0
    b1_gradient = a2.sum(axis=0)
    W1_gradient = np.dot(train_x.T, a2)

    # パラメータ更新
    W1 = W1 - LEARNING_RATE * W1_gradient
    W2 = W2 - LEARNING_RATE * W2_gradient
    b1 = b1 - LEARNING_RATE * b1_gradient
    b2 = b2 - LEARNING_RATE * b2_gradient

# 結果表示

# 最終訓練データのL値
L = cross_entropy_error(forward(train_x), train_t)
print(L)

# テストデータの結果
test_y = forward(test_x)
print((test_y.argmax(axis=1) == test_t.argmax(axis=1)).sum(), '/', 150 - TRAIN_DATA_SIZE)
