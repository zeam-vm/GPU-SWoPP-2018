# はじめに

GPGPU技術\cite{Nickolls:2008:SPP:1401132.1401152}が活発に研究開発されてきた．現在主流となっているGPUのアーキテクチャはSIMD(単一命令複数データ)モデルに基づいていて超並列計算に向いているため，システムを並列に計算させることができる．

GPGPUの研究開発が注目されてきた理由の1つは，機械学習が社会実装に適用されることが一般的になり，より高いパフォーマンスが要求されるようになってきたからである．機械学習で用いられるプログラミング言語のデファクトスタンダードはPython\cite{Python}である．しかし，CUDA\cite{CUDA}やCuPy\cite{CuPy}，TensorFlow\cite{TensorFlow}，Keras\cite{Keras}といったPythonベースの現行のGPGPUライブラリには次の2つの問題点がある．

1. **性能の問題:** Python \cite{Python} は，NumPy\cite{NumPy}やCuPy\cite{CuPy}のようなネイティブライブラリを除けば単一スレッドのインタプリタ言語であるため，基本的に低いパフォーマンスである．このことにより，システムがネイティブライブラリで処理する前にPythonインタプリタで大量のデータを処理するときに深刻な問題となる．
2. **設定の問題:** PythonがGPUを使えるようにするネイティブライブラリを設定するのが難しい．これらはインストールや設定がとても面倒である．さらに，NVIDIAのGPUしかサポートされない．

このことから，我々は機械学習の目的でElixir\cite{Elixir}を採用する．その理由は，Elixirはとても良い並列処理性能を発揮するからである\cite{Elixir16}．しかし，現在のElixirはGPGPUをサポートしない．そこで，我々はElixirにGPGPUサポートを実装することを試みた．

我々が提案するHastegaは，MapReduce モデル\cite{Dean:2008:MSD:1327452.1327492}に似ているElixirのFlow\cite{Flow}を用いたプログラミングスタイルがGPGPUのコードに容易に変換できるという仮説に基づいている．また我々はGPUを操作するGPGPUコードをElixirとRust\cite{Rust}で記述しRustler\cite{Rustler}を用いて実装した．

本稿の以下の構成は次のとおりである: 第2章ではElixirコードからGPUを駆動するコードに変換する中核となるアイデアとHastegaの実装について提案する．第3章ではインストールと設定のプロセスについて示す．第4章では我々が行なった実験の実行環境と，ベンチマークについての記述，結果と考察を示す．第5章では本稿をまとめて将来課題について述べる．

# Hastegaの方針と実装

MapReduceモデル\cite{Dean:2008:MSD:1327452.1327492}に基づくFlow\cite{Flow}を用いた並列プログラミングスタイル(\figref{fig:flow})は，GPUに採用されているSIMD(単一命令複数データ)アーキテクチャにそのまま適合する．すなわち，パイプライン演算子で繋がれている関数群 `foo |> bar` は「単一命令」に，範囲`1..1000`によって生成されるリストは「複数データ」に，それぞれ適合する．このことから，Flowで書かれたElixirのコードはGPUで実行できるコードに容易に変換できる．

\begin{figure}[t]
\setbox0\vbox{
{\small
\begin{verbatim}
1..1000
  |> Flow.from_enumerable()
  |> Flow.map(foo)
  |> Flow.map(bar)
  |> Enum.to_list 
\end{verbatim}
}
}
\centerline{\fbox{\box0}}
\caption{Flowを使ってリストを操作するElixirコード}
\ecaption{Elixir code of list manipulation using Flow.}
\label{fig:flow}
\end{figure}

**(TODO: GPUのコード例)**

\figref{fig:Hastega-arch}にHastegaのアーキテクチャを示す．我々はRust\cite{Rust}でネイティブコードを記述し，Rustler\cite{Rustler}とocl\cite{ocl}を用いた．このことによりとても設定が容易になる．プログラミング言語とOpenCL\cite{OpenCL}とHastegaをそれぞれ1行程度のコマンドでインストールするだけである．これは，Python\cite{Python}や
CUDA\cite{CUDA}と，CuPy\cite{CuPy}のような関連ライブラリと比べて大きな優位性を持っている．

\begin{figure}[t]
\includegraphics{Hastega-arch}
\caption{Hastegaアーキテクチャ}
\ecaption{The Hastega Architecture}
\label{fig:Hastega-arch}
\end{figure}

我々の実装はGitHubに公開されている\footnote{LogisticMap: Benchmark of Logistic Map using integer calculation and Flow, available at https://github.com/zeam-vm/logistic\_map}．


# Hastegaの特長: 容易な設定

\tabref{setting}にGCE(Google Compute Engine)\cite{GCE}上でのインストール・設定プロセスの比較を示す．

Hastegaのインストールはビルドツールが自動で設定してくれるので煩雑ではない．必要なことは，OpenCL\cite{OpenCL}をインストールすること，Elixir\cite{Elixir}とRust\cite{Rust}をインストールして設定すること，Hastegaをインストールすることだけである．


CuPy\cite{CuPy}におけるCUDA\cite{CUDA}もしくはOpenCL\cite{OpenCL}のインストールは，Hastegaと比べてより多くの作業手順を必要とする．その理由は，Cupyが古いバージョンのCUDA\cite{CUDA}を必要とするからである．この解決方法を知るためにStack Overflow\cite{StackOverflow}のようなQ\&Aサイトを調べ上げる必要があった．

Hastegaの欠点は，Elixir\cite{Elixir}とRust\cite{Rust}という2つのプログラミング言語のインストールと設定を必要とする点である．


\begin{table}[t]
\centering
\caption{GCE上でのインストール・設定プロセスの手順の比較}
\ecaption{Comparison of Steps of Installation and Setting Processes in GCE}
\label{setting}
{\small
\begin{tabular}{lrr}
                               & \multicolumn{1}{l}{CuPy} & \multicolumn{1}{l}{Hastega} \\ \hline
CUDAもしくはOpenCLのインストール   & 4                        & 1                           \\
プログラミング言語のインストール     & 0                        & 4                           \\
ライブラリのインストール           & 2                        & 1                           \\ \hline
\end{tabular}
}
\end{table}

# パフォーマンス評価

## 評価環境

我々は素体によるロジスティック写像\cite{Miyazaki14}をベンチマークとして採用した．これは下記のような漸化式で表される．

\begin{displaymath}
  X_{i+1} = \mu_p X_i (X_i + 1) \bmod p
\end{displaymath}

これを採用した理由は，GPUのアセンブリコードに容易に変換できる整数演算を用いて負荷の高い計算をすることができるためである．

我々はMac Pro (Mid 2010)とGCE\cite{GCE}という2つの環境で評価した．

1. Mac Pro (Mid 2010) は1つの2.8GHzのクアッドコア Intel Xeon のCPUと，16GB のメモリ，そして1024MBのメモリを持つATI Radeon HD 5770で構成される．

2. GCEの設定は次のとおりである．
	* Machine type: custom (8 vCPUs，16GBメモリ)
	* CPU platform: Intel Broadwell
	* GPU: NVIDIA Tesla K80 (x1)
	* Zone: us-west1-b

\tabref{versions}に用いたソフトウェアのバージョンを示す．

\begin{table*}[t]
\centering
\caption{用いたソフトウェアのバージョン}
\ecaption{Versions of Software}
\label{versions}
{\small
\begin{tabular}{l|ll}
                       & Mac Pro (Mid 2010)     & GCE                                      \\ \hline
OS                     & Sierra 10.12.6         & ubuntu 16.04                             \\
Elixir \cite{Elixir}   & 1.6.6 (OTP 21)         & 1.6.6 (OTP 21)                           \\
Flow \cite{Flow}       & 0.14                   & 0.14                                     \\
Rust \cite{Rust}       & 1.27.0                 & 1.27.0                                   \\
OpenCL \cite{OpenCL}   & 1.2                    & 1.2                                      \\
Rustler \cite{Rustler} & 0.17.1                 & 0.17.1                                   \\
ocl \cite{ocl}         & 0.18                   & 0.18                                     \\
rayon \cite{rayon}     & 1.0                    & 1.0                                      \\
scoped-pool \cite{scoped-pool} & 1.0.0          & 1.0.0                                    \\ \hline
Python \cite{Python}   & 3.6.0 (Anaconda 4.3.0) & 3.5.2                                    \\
CUDA \cite{CUDA}       & N/A                    & 9.0 (in case of using CuPy), 9.2 (other) \\
NumPy \cite{NumPy}     & 1.11.3                 & 1.14.3                                   \\
CuPy \cite{CuPy}       & N/A                    & 4.1.0
\end{tabular}
}
\end{table*}

## ベンチマーク

本実験では下記のベンチマークを用いた:

* Elixir\_recursive: Elixir\cite{Elixir}のみで書かれている．ロジスティック写像の計算は10回の再帰呼出しで実装されている．
* Elixir\_inlining: Elixirのみで書かれている．ロジスティック写像の計算はFlowの中にインライン展開されている．
* Rustler\_CPU: ElixirとRust\cite{Rust}で書かれており，Rustler\cite{Rustler}を用いている．ロジスティック写像はRustで書かれたネイティブコードでCPUで計算している．scoped-pool\cite{scoped-pool}によるスレッドプールを用いた非同期NIF(Native Implemented Function)呼出しを用いており，単一スレッドで動作する．
* Rustler\_CPU\_multi: Rustler\_CPUと同様であるが，ロジスティック写像をrayon\cite{rayon}による複数スレッドで計算する点が異なる．ただし，実装上の制約でrayonのスレッドプールは有効にできなかった．
* Rustler\_GPU: ElixirとRustで書かれており，Rustlerを用いている．ロジスティック写像はRustとocl\cite{ocl}を用いて書いたネイティブコードでOpenCL\cite{OpenCL}経由でGPUで計算されている．
* Empty: 実行効率の観点で比較するためのダミーベンチマークである．ElixirのリストとRustのベクターの返還を含むが，ロジスティック写像の計算は含まない．
* Rust\_CPU: Rustのみで書かれている．ロジスティック写像は単一スレッドのCPUで計算されている．
* Rust\_CPU\_multi: Rust\_CPUと同様であるが，ロジスティック写像はrayon\cite{rayon}を用いて複数スレッドで実行されている．かつ，rayonのスレッドプールが有効である．
* Rust\_GPU: Rustだけで書かれている．ロジスティック写像はoclを用いてOpenCL経由でGPUで計算されている．
* Python\_CPU: Python\cite{Python}とNumPy\cite{NumPy}で書かれている．
* Python\_GPU: Python\cite{Python}とCuPy\cite{CuPy}で書かれており，GPUで実行する．

## 評価の結果と考察

\tabref{result}にベンチマークの結果を示す．

* Elixirのみ(Elixir\_recursive と Elixir\_inlining)と比べて，Rustler\_GPU は4.43--8.23倍，Rustler\_CPU\_multiは5.68--6.97倍高速である．
* Rustler\_CPU\_multiとRustler\_GPUは，Mac Pro (Mid 2010)とGCEで逆になっている．この理由は，rayonのスレッドプールがないことで，Linuxではパフォーマンスが悪化するためである．
* Rustler\_GPUとEmptyの差の比は22.2--27.6％である．これがElixirのリストとRustのベクターを変換するオーバーヘッド等を除いた実質的な実行時間であると考えられる．
* Python\_CPUと比べて，Elixirのみは1.17--1.68倍，Rustler\_GPUは7.43--9.64倍高速である．
* Python\_GPUと比べて，Rustler\_GPU は3.67倍高速である．
* Rustler\_CPUとRust\_CPUの差の比は62.0--70.0％，Rustler\_GPUとRust\_GPUの差の比は32.2--35.2％である．これはEalang VMのオーバーヘッドであると考えられる．
* Rustler\_GPUと比べて，Rust\_GPUは1.48--1.54倍高速である．これが潜在的な最適化の余地であると考えられる．

\begin{table*}[t]
\centering
\caption{ベンチマーク結果}
\ecaption{The result of the benchmarks}
\label{result}
{\small
\begin{tabular}{lll|r|r|}
           &                  &              & \multicolumn{1}{l|}{Mac Pro (Mid 2010)} & \multicolumn{1}{l|}{GCE}              \\
           &                  &              & \multicolumn{1}{l|}{2.8GHz Quad-Core Intel Xeon} & \multicolumn{1}{l|}{Intel Broadwell vCPU:8}           \\
           &                  &              & \multicolumn{1}{l|}{ATI Radeon HD 5770} & \multicolumn{1}{l|}{NVIDIA Tesla K80} \\ 
           &                  &              &  (秒)        & (秒) \\ \hline
Elixir\_recursive             & Elixir           & 再帰呼出し       & 12.177           & 9.674            \\
Elixir\_inlining              & Elixir           & インライン展開    & 10.579           & 8.075            \\
Rustler\_CPU                  & Elixir / Rustler & CPU            & 7.691            & 6.098            \\
\rowcolor[HTML]{C0C0C0}
Rustler\_CPU\_multi           & Elixir / Rustler & CPU            & 1.748            & 1.422              \\
\rowcolor[HTML]{C0C0C0}
Rustler\_GPU                  & Elixir / Rustler & OpenCL (GPU)   & 2.388            & 1.176            \\
Empty                         & Elixir / Rustler & empty          & 1.859            & 0.852            \\
Rust\_CPU                     & Rust             & CPU            & 2.926            & 1.829            \\
\rowcolor[HTML]{C0C0C0}
Rust\_CPU\_multi              & Rust             & CPU            & 0.669            & 0.374             \\
\rowcolor[HTML]{C0C0C0}
Rust\_GPU                     & Rust             & OpenCL (GPU)   & 1.546            & 0.797           \\
Python\_CPU                   & Python           & NumPy (CPU)    & 17.749           & 11.341          \\
Python\_GPU                   & Python           & CuPy (GPU)     & N/A              & 4.316           \\
\end{tabular}
}
\end{table*}

