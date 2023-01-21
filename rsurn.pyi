from typing import List, Tuple

class Gene:
    """遺伝子の情報: 環境の振る舞い方を定義する"""

    def __new__(cls: type["Gene"], rho: int, nu: int, recentness: float, friendship: float) -> "Gene":
        """遺伝子の情報: 環境の振る舞い方を定義する

        rho, nu は主にモデル全体の挙動を定義し、recentness, friendship はエージェントの振る舞いを定義する。

        Parameters
        ----------
        rho : int
            優先的選択性の強さの度合い。1以上の整数を指定。
        nu : int
            新規性への積極性の度合い。1以上の整数を指定。
        recentness : float
            最近やり取りしたエージェントを紹介する度合い。任意の実数を指定。
            値が大きいほど最近やり取りしたエージェントを高い確立で紹介する。
            マイナス値の場合、昔やり取りした(最近やり取りしていない)エージェントが優先して紹介される。0の場合ランダム。
        friendship : float
            親密なエージェントを紹介する度合い。
            任意の実数を指定。値が大きいほど親密な(自身の壺に多く含まれる)エージェントを高い確立で紹介する。
            マイナス値の場合、親密でないエージェントが優先して紹介される。0の場合ランダム。

        recentness, friendship がともに 0 と指定された場合、完全にランダムな挙動となる。

        Returns
        -------
        Gene
            生成された遺伝子。Environment クラスの生成に用いる。
        """

class Environment:
    """環境: エージェントが振る舞うプラットフォーム"""

    history: List[Tuple[int, int]]
    """生成された (caller, callee) 組の履歴"""

    def __new__(cls: type["Environment"], gene: Gene) -> "Environment":
        """環境: エージェントが振る舞うプラットフォーム

        Parameters
        ----------
        gene : Gene
            遺伝子の情報。Geneクラスのドキュメントを参照のこと。

        Returns
        -------
        Environment
            生成された環境。
        """
        ...
    def get_caller(self) -> int:
        """環境から caller を取得する

        Returns
        -------
        int
            callerエージェントのID。IDは0以上の整数値。
        """
        ...
    def get_callee(self, caller: int) -> int:
        """環境中に存在する caller から callee を取得する

        Parameters
        ----------
        caller : int
            callerエージェントのID。IDは0以上の整数値。

        Returns
        -------
        int
            calleeエージェントのID。IDは0以上の整数値。
        """
        ...
    def interact(self, caller: int, callee: int) -> None:
        """環境上で caller と callee に相互作用を起こさせる

        Parameters
        ----------
        caller : int
            callerエージェントのID。IDは0以上の整数値。Environment#get_caller を参照。
        callee : int
            calleeエージェントのID。IDは0以上の整数。Environment#get_callee を参照。
        """
        ...
