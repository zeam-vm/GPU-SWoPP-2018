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