# rsurn
エージェントベースのポリアの壺モデルのRust実装 + Pythonモジュール

## 変更履歴
### v0.2.2
- 重みがすべて0になる状況が生まれ、一度の同じ要素が複数紹介されてしまうバグを修正

### v0.2.1
- 親密度が負の場合モデルを実行できないバグを修正

### v0.2.0
- 親密度と最近度の遺伝子を追加
- モデル詳細: https://websci-lab.slack.com/archives/C04E9DB2L59/p1674292619405879

### v0.1.0
- Ubaldiのエージェントベースのポリアの壺モデルを再実装
- 戦略は WSW (壺内からランダムに nu+1 個選択) に固定
- パラメータは ρ, ν のみ
- Julia版の約2倍の性能


## 依存関係
Pythonから使用する場合でもビルドのためにRustが必要です。

https://www.rust-lang.org/ja/tools/install

## インストール
### Pythonで使用する場合
pip や poetry を使ってこの GitHub レポジトリから直接インストールしてください。

```sh
pip install git+https://github.com/tsukuba-websci/rsurn.git
# or
poetry add git+https://github.com/tsukuba-websci/rsurn.git
# or any other package manager you like :)
```

Python では Type Hint も利用できます。Visual Studio Code の Python 拡張機能や mypy の利用を推奨します。

### Rustで使用する場合
cargo を使ってこの GitHub レポジトリから直接インストールしてください。

```sh
cargo add --git https://github.com/tsukuba-websci/rsurn.git
```

## 使い方
次のサンプルプログラムを参考にしてください。

### Python
```py
from rsurn import Gene, Environment

gene = Gene(3, 4, 0.5, -0.5) # rho, nu, recentness, friendship の順に指定
env = Environment(gene)

for _ in range(1000):
    caller = env.get_caller()
    callee = env.get_callee(caller)
    env.interact(caller, callee)

# List[Tuple[caller_id: int, callee_id: int]] 型の生成データを出力
print(env.history)
```

### Rust

```rust
let gene = Gene {
    rho: 3,
    nu: 4,
    recentness: 0.5,
    friendship: -0.5
};
let mut env = Environment::new(gene);

for _ in 0..10000 {
    let caller = env.get_caller().unwrap();
    let callee = env.get_callee(caller).unwrap();
    env.interact(caller, callee);
}

println!("{:#?}", env.history);
```

## ToDo
- [ ] [lib.rs](/src/lib.rs) のリファクタリング
  - [ ] メソッド名の見直し
  - [ ] コメントの付与
- [ ] GitHub Actions によるテストの自動化
