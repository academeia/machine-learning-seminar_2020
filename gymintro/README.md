# 機械学習勉強会2.0ー強化学習入門

## Abstract
強化学習のシミュレーション環境 **OpenAI Gym** を通して，強化学習の基礎について学びます．

## 0. Introduction

### 0.1. 強化学習とは？
**強化学習 (reinforcement learning)** とは，最適な意思決定の方針を求めることを
目的とする学問分野であり，「教師あり学習」や「教師なし学習」と並ぶ機械学習の一分野です．

```
機械学習
  ├── 教師あり学習 (例：分類，回帰)
  ├── 教師なし学習 (例：クラスタリング，次元削減)
  └── 強化学習 [今回のテーマ]
```

強化学習には他の機械学習分野にはない **報酬 (reward)** という概念があり，
獲得する報酬の期待値を最大にするような **行動 (action)** を決める方策を学習する，
というのが特徴的です．

### 0.2. OpenAI Gym
OpenAI Gym とは，人工知能を研究する非営利団体である OpenAI が作成した，
強化学習のシミュレーション用プラットフォームです．

* OpenAI Gym 公式サイト:
 [https://gym.openai.com](https://gym.openai.com/)
* OpenAI Gym GitHub:
 [https://github.com/openai/gym](https://github.com/openai/gym)

本稿では，OpenAI Gym が提供する環境「山登りゲーム」 (`MountainCar-v0`) を通して，
強化学習の一種である **Q学習 (Q-learning)** の理論と実装について解説します．

### 0.3. 強化学習の枠組み
強化学習は一般的に以下の図のようなシステムとの相互作用から学習します．

```
       a: action
    ┌ ─ ─ ─ ─ ─ ─ ┐
┌ ─ ┴ ─ ┐ ┌ ─ ─ ─ v ─ ─ ┐
│ agent │ │ environment │
└ ─ ^ ─ ┘ └ ─ ─ ─ ┬ ─ ─ ┘
    └ ─ ─ ─ ─ ─ ─ ┘
 r: reward, s': next state
```

__基本用語__

* **状態 (state)**: システムの状態．
* **行動 (action)**: エージェントの行動．
* **報酬 (reward)**: 行動の結果の良し悪しを測る量．数学的には，
{状態} * {行動} * {状態} 上の実数値関数として定義される．
* **環境 (environment)**: 制御対象となるシステムのこと．ゲーム環境．
エージェントが行動を与えると (確率的に) 状態が変化し，それに応じて報酬が得られる．
数学的には，状態の集合，行動の集合，報酬関数，状態遷移確率関数の4つ組のことをいう．
この4つ組を **マルコフ決定過程 (Markov decision process: MDP)** という．
* **エージェント (agent)**: 制御器もしくは意思決定者のこと．強化学習の目的は，
状態に対して行動を返すような関数 (これを **政策 (policy)** という) を学習することである．

さて，強化学習を実装するには，まずは制御したい環境の内容を考えて，
その環境を実装する必要があります．また，その状態を学習中でも視覚的に
分かりやすく表示する機能があると嬉しいですよね．そこで役に立つのが今回の OpenAI Gym です．
OpenAI Gym は環境周辺のあらゆる機能を提供してくれます．

## 1. OpenAI Gym の使い方
では早速 OpenAI Gym を使ってみましょう．

### 1.1. インストール
Python3 のパッケージ管理ツール pip3 を用いて OpenAI Gym をインストールします．

```bash
pip3 install gym
```

2020年4月30日現在，公開されている OpenAI Gym の最新バージョンは 0.17.1 となっています．

```shell-session
$ pip3 show gym
Name: gym
Version: 0.17.1
Summary: The OpenAI Gym: A toolkit for developing and comparing your reinforcement learning agents.
Home-page: https://github.com/openai/gym
Author: OpenAI
Author-email: gym@openai.com
License: UNKNOWN
Location: /usr/local/lib/python3.7/site-packages
Requires: scipy, cloudpickle, numpy, six, pyglet
```

### 1.2. 環境データ
まずは環境 (environment) を作成してみましょう．
`gym.make()` の引数に環境名 (ゲーム名) を渡して，環境インスタンス `env` を生成します．

```python
import gym
env = gym.make('MountainCar-v0')
```

生成可能な環境名の一覧を取得するには次のようにしてください．

```python
from gym import envs
for spec in envs.registry.all():
  print(spec.id)
```

それぞれの詳細については公式サイトを参照してください．

* OpenAI Gym Environments: [https://gym.openai.com/envs](https://gym.openai.com/envs/)

環境の初期化をするには `reset()` メソッドを実行します．
この戻り値 `observation` は初期状態における観測データを表しています．
詳細については次の節で説明します．

```python
observation = env.reset()
```

現在の環境を描画するには `render()` メソッドを実行してください．

```python
env.render()
```

すると，次のような山登りゲームの初期状態画面が出力されます．

![初期状態](https://github.com/academeia/machine-learning-seminar_2020/blob/images/initial_state.png "初期状態")

### 1.3. 観測データ
今回の `MountainCar-v0` の場合は次のような配列が観測データとなります．

```python
print(observation)
>> [-0.5497433  0.        ]
```

観測データの最大値と最小値は次のようにして得られます．

```python
# Max
print(env.observation_space.high)
>> [0.6  0.07]
# Min
print(env.observation_space.low)
>> [-1.2  -0.07]
```

| 成分 | 0 (車の位置) | 1 (車の速度) |
|:---:|:----:|:-----:|
| Max | 0.6  | 0.07  |
| Min | -1.2 | -0.07 |

今回の環境では，車の位置が 0-index, 車の速度が 1-index に格納されます．
なお，観測データのそれぞれの意味についてはゲーム環境によって異なります．
詳細については GitHub Wiki を参照してください．

* GitHub Wiki: [https://github.com/openai/gym/wiki](https://github.com/openai/gym/wiki)

### 1.4. 行動
では実際にゲーム環境を操作してみましょう．
`env.step()` の引数に `action` を渡すことで行動できます．
今回の `MountainCar-v0` の場合は次のような操作が可能です．

| 値 | 内容 |  
|:-:|:-:|
| 0 | 左へ押す |
| 1 | 何もしない |
| 2 | 右へ押す |

なお，行動の値とそれぞれの意味についてはゲーム環境によって異なります．
詳細については GitHub Wiki を参照してください．

* GitHub Wiki: [https://github.com/openai/gym/wiki](https://github.com/openai/gym/wiki)

それでは，試しに `action=0` としてみましょう．

```python
action = 0
env.step(action) # -> observation, reward, done, info
```

すると，4つの戻り値 `observation`, `reward`, `done`, `info` が得られます．
それぞれの詳細は次の通りです．

* `observation`: 行動 `action` により変化した環境の観測データ
* `reward`: 行動 `action` による報酬を表す (今回はゴール以外全て `-1.0` で設定)
* `done`: ゲームが終了したか否かを表す (bool 値)
* `info`: 詳細情報 (今回は存在しない)

今度は，試しに200回 `action=2` (右へ押す) を連続で実行してみましょう．

```python
action = 2
env.reset()
for _ in range(200):
  env.step(action)
  env.render()
```

![「右に押す」を200回繰り返す](https://github.com/academeia/machine-learning-seminar_2020/blob/images/right_pushing.gif "「右に押す」を200回繰り返す")

今回の `MountainCar-v0` は一筋縄ではうまくいかず，
毎回「右へ押す」を選択してもゴール前の登り坂で減速してしまい，
たどり着くことはできません．そこで，ある程度「左へ押す」を **工夫して** 活用することで，
勢いをつけてから坂を登り，ゴールを目指す必要があります．
その工夫部分を試行錯誤で学習するのが今回の強化学習なのです．

## 2. Q-Learning
本稿では，強化学習の代表的な手法の一つである **Q学習 (Q-learning)**
を実装することで，ゲーム `MountainCar-v0` をクリアすることを目指します．

### 2.1. 理論
(加筆中)

### 2.2. 実装
それではQ学習を実装してみましょう．
まずは必要なモジュールの import とハイパーパラメータを設定します．

```python
#!/usr/bin/env python3
import gym
import numpy as np

A = 3 # the number of action
N = 50 # the number of one of status
lr = 0.2 # the learning rate
gamma = 0.99 # the time discount rate
epsilon = 0.002 # used in the epsilon-greedy method
episode_num = 10000 # the number of episodes
step_num = 200 # the number of steps in one episode
```

今回の `MountainCar-v0` では実行できる行動が `A=3` 種類あります．

また，観測データにおける車の位置と速度は連続値であるので，
実装のためにそれぞれ適当な `N=50` 個の離散値に変換します．

```python
# transform observation to status in {0, ..., N-1}
def get_status(env, observation):
  env_min = env.observation_space.low
  env_max = env.observation_space.high
  env_dx = (env_max - env_min) / N
  p = int((observation[0] - env_min[0]) / env_dx[0]) # position
  v = int((observation[1] - env_min[1]) / env_dx[1]) # velocity
  return p, v
```

次に，Q学習におけるQテーブルを初期化します．

```python
# initialize q-table
q_table = np.zeros((N, N, A))
```

次に，Qテーブルを更新する関数を定義しましょう．

```python
def update_q_table(q_table, action, observation, next_observation, reward):
  # Q(s, a)
  p, v = get_status(env, observation)
  q_value = q_table[p][v][action]

  # max Q(s', a')
  next_p, next_v = get_status(env, next_observation)
  next_max_q_value = max(q_table[next_p][next_v])

  # update q-table
  q_table[p][v][action] = q_value + lr * (reward + gamma * next_max_q_value - q_value)

  return q_table
```

次に，epsilon-greedy 法を実装しましょう．
基本的には価値関数 `Q(s,a)` が最大となる行動を返しますが，
一定確率 `epsilon` でランダムに行動する点に注意してください．

```python
def get_action(env, q_table, observation, epsilon=epsilon):
  if np.random.uniform(0, 1) > epsilon:
    p, v = get_status(env, observation)
    action = np.argmax(q_table[p][v])
  else:
    action = np.random.choice(range(A))
  return action
```

最後に，各 episode における学習を実装しましょう．

```python
def one_episode(env, q_table, init_observation, rewards, episode):
  # initialization
  total_reward = 0
  observation = init_observation

  for _ in range(step_num):
    # choose an action by epsilon-greedy method
    action = get_action(env, q_table, observation)

    # move the car, get the next observation and reward
    next_observation, reward, done, _info = env.step(action)

    # update q-table
    q_table = update_q_table(q_table, action, observation, next_observation, reward)

    # update observation and `total_reward`
    observation = next_observation
    total_reward += reward

    if done: # if True, then finish one episode
      rewards.append(total_reward)
      if episode % 100 == 0:
        print("episode: {}, total_reward: {}".format(episode, total_reward))
      break

  return q_table, rewards
```

以上を `__main__` にまとめると次のようになります．

```python
if __name__ == '__main__':
  env = gym.make('MountainCar-v0')
  rewards = []

  # initialize q-table
  q_table = np.zeros((N, N, A))

  # learning
  for episode in range(episode_num):
    init_observation = env.reset()
    q_table, rewards = one_episode(env, q_table, init_observation, rewards, episode)

  # show result of learning
  for _ in range(step_num):
    # initialization
    observation = env.reset()

    # choose an action by taking argmax of q-table
    action = get_action(env, q_table, observation, epsilon=-1)

    # move the car, get the next observation and reward
    observation, _reward, done, _info = env.step(action)

    # show environment
    env.render()

    if done:
      env.close()
      break
```

![結果](https://github.com/academeia/machine-learning-seminar_2020/blob/images/result.gif "結果")

![報酬合計の推移](https://github.com/academeia/machine-learning-seminar_2020/blob/images/rewards.png "報酬の推移")


解説は以上となります．他の環境も是非試してみてください．

## 3. 参考文献

* 教科書
  * 森村哲郎『強化学習』(機械学習プロフェッショナルシリーズ)
  * S. Sutton, G. Barto "Rainforcement Learning An Introduction" (The MIT Press)


* Qiita
  * OpenAI Gym 入門 [[link](https://qiita.com/ishizakiiii/items/75bc2176a1e0b65bdd16)]
  * ゼロから Deep まで学ぶ強化学習 [[link](https://qiita.com/icoxfog417/items/242439ecd1a477ece312)]
  * DQN の生い立ち + Deep Q-Network を Chainer で書いた [[link](https://qiita.com/Ugo-Nama/items/08c6a5f6a571335972d5)]
