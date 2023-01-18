from typing import List, Tuple

class Gene:
    """遺伝子の情報: 環境の振る舞い方を定義する"""

    def __new__(cls: type["Gene"], rho: int, nu: int) -> "Gene":
        """遺伝子の情報: 環境の振る舞い方を定義する

        Parameters
        ----------
        rho : int
            優先的選択性の強さの度合い。1以上の整数を指定。
        nu : int
            新規性への積極性の度合い。1以上の整数を指定。

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
