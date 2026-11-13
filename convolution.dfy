include "basic_arithmetic.dfy"
include "linear_algebra.dfy"
include "neural_networks.dfy"

/**
  * Convolutional layers (CNN extension). A zero-padded convolution is linear, so
  * it is a matrix: its banded doubly block Toeplitz matrix (Delattre et al. 2024,
  * Sec. V-B). A CNN is therefore a CAV 2025 NeuralNetwork, and the definitions of
  * robustness and margin Lipschitz bounds apply to it unchanged. The certifier
  * uses these matrices only in ghost code, so they are never built at runtime.
  */
module Convolution {
  import opened BasicArithmetic
  import opened LinearAlgebra
  import opened NeuralNetworks

  /**
    * c(cOut, k, stride, pad) on an input of cIn channels of n x n, without bias,
    * as make_certifier_format.py writes it: conv(cIn,n,cOut,k,stride,pad).
    * K[o][c][a][b] is the weight for output channel o, input channel c, kernel
    * row a and kernel column b. It computes the same cross-correlation as Keras'
    * Conv2D on the zero-padded input.
    */
  datatype ConvLayer = ConvLayer(K: seq<seq<seq<seq<real>>>>, cIn: nat, n: nat,
                                 cOut: nat, k: nat, stride: nat, pad: nat)

  /** Positive sizes, a k x k kernel for each pair of channels, and an output. */
  predicate ValidConv(L: ConvLayer) {
    L.cIn > 0 && L.n > 0 && L.cOut > 0 && L.k > 0 && L.stride > 0 &&
    L.k <= L.n + 2 * L.pad &&
    |L.K| == L.cOut &&
    forall o | 0 <= o < L.cOut ::
      |L.K[o]| == L.cIn &&
      forall c | 0 <= c < L.cIn ::
        |L.K[o][c]| == L.k &&
        forall a | 0 <= a < L.k :: |L.K[o][c][a]| == L.k
  }

  /** Output height (= width), as doitlib.conv_output_size computes it. */
  function OutSize(L: ConvLayer): (m: nat)
    requires ValidConv(L)
    ensures m >= 1
  {
    (L.n + 2 * L.pad - L.k) / L.stride + 1
  }

  /**
    * The entry at row (o, y, x) and column (c, i, j). Output (o, y, x) adds up
    * K[o][c][a][b] times the padded input at row y*stride + a and column
    * x*stride + b, which is input (c, i, j) with i = y*stride + a - pad and
    * j = x*stride + b - pad. Padding zeros add nothing, so they get no column.
    */
  function ToeplitzEntry(L: ConvLayer, o: nat, y: nat, x: nat,
                         c: nat, i: nat, j: nat): real
    requires ValidConv(L)
    requires o < L.cOut && c < L.cIn
  {
    var a := i + L.pad - y * L.stride;
    var b := j + L.pad - x * L.stride;
    if 0 <= a < L.k && 0 <= b < L.k then L.K[o][c][a][b] else 0.0
  }

  /**
    * The layer's banded doubly block Toeplitz matrix: ToeplitzMatrix(L)·v is the
    * layer's output, before ReLU, for input v. Vectors are listed channel by
    * channel, then row by row: input (c, i, j) has index c*n*n + i*n + j, and
    * output (o, y, x) has index o*m*m + y*m + x, where m = OutSize(L).
    * Only specifications and proofs use it, so the certifier never builds it.
    */
  function ToeplitzMatrix(L: ConvLayer): (T: Matrix)
    requires ValidConv(L)
    ensures Rows(T) == L.cOut * (OutSize(L) * OutSize(L))
    ensures Cols(T) == L.cIn * (L.n * L.n)
  {
    var m := OutSize(L);
    MulPos(m, m);
    MulPos(L.cOut, m * m);
    seq(L.cOut * (m * m), r requires 0 <= r < L.cOut * (m * m) => ToeplitzRow(L, r))
  }

  /** Row r = o*m*m + y*m + x of ToeplitzMatrix(L). */
  function ToeplitzRow(L: ConvLayer, r: nat): (row: Vector)
    requires ValidConv(L)
    requires r < L.cOut * (OutSize(L) * OutSize(L))
    ensures |row| == L.cIn * (L.n * L.n)
  {
    var m := OutSize(L);
    var nn := L.n * L.n;
    MulPos(m, m);
    MulPos(L.n, L.n);
    MulPos(L.cIn, nn);
    DivBound(r, L.cOut, m * m);
    seq(L.cIn * nn, col requires 0 <= col < L.cIn * nn =>
      DivBound(col, L.cIn, nn);
      ToeplitzEntry(L, r / (m * m), (r / m) % m, r % m,
                    col / nn, (col / L.n) % L.n, col % L.n))
  }

  /** A CNN layer: a dense layer's weight matrix (as in CAV 2025), or a conv layer. */
  datatype CNNLayer = Dense(W: Matrix) | Conv(C: ConvLayer)

  predicate ValidLayer(l: CNNLayer) {
    l.Conv? ==> ValidConv(l.C)
  }

  /** The number of rows (outputs) of the layer's matrix, without building it. */
  function LayerRows(l: CNNLayer): nat
    requires ValidLayer(l)
  {
    match l
    case Dense(W) => Rows(W)
    case Conv(C) => C.cOut * (OutSize(C) * OutSize(C))
  }

  /** The number of columns (inputs) of the layer's matrix, without building it. */
  function LayerCols(l: CNNLayer): nat
    requires ValidLayer(l)
  {
    match l
    case Dense(W) => Cols(W)
    case Conv(C) => C.cIn * (C.n * C.n)
  }

  /** The layer's matrix. Ghost, so that the certifier never builds it. */
  ghost function LayerMatrix(l: CNNLayer): (M: Matrix)
    requires ValidLayer(l)
    ensures Rows(M) == LayerRows(l) && Cols(M) == LayerCols(l)
  {
    match l
    case Dense(W) => W
    case Conv(C) => ToeplitzMatrix(C)
  }

  /**
    * Valid layers whose sizes chain, ending in a dense layer (d(10) in the
    * architectures), so that the certifier has the last layer's weights at
    * runtime. Compilable, so the certifier can check it on the parsed layers.
    */
  predicate IsCNN(ls: seq<CNNLayer>) {
    |ls| > 0 &&
    (forall i | 0 <= i < |ls| :: ValidLayer(ls[i])) &&
    ls[|ls|-1].Dense? &&
    forall i | 0 <= i < |ls| - 1 :: LayerRows(ls[i]) == LayerCols(ls[i+1])
  }

  /**
    * The CNN as a CAV 2025 NeuralNetwork, each conv layer replaced by its Toeplitz
    * matrix: NN(CNNToNeuralNetwork(ls), v) is the CNN's output for input v.
    */
  ghost function CNNToNeuralNetwork(ls: seq<CNNLayer>): (n: NeuralNetwork)
    requires IsCNN(ls)
    ensures |n| == |ls|
    ensures forall i | 0 <= i < |ls| :: n[i] == LayerMatrix(ls[i])
    ensures n[|n|-1] == ls[|ls|-1].W
  {
    seq(|ls|, i requires 0 <= i < |ls| => LayerMatrix(ls[i]))
  }

  /* ============ The Toeplitz matrix computes the zero-padded Conv2D ============ */

  /**
    * An image of c channels of n x n pixels: I[ch][i][j] is pixel (i, j) of
    * channel ch, as in a Keras input or output with the channels moved first.
    */
  predicate IsImage(I: seq<seq<seq<real>>>, c: nat, n: nat) {
    |I| == c &&
    forall ch | 0 <= ch < c :: |I[ch]| == n &&
                               forall i | 0 <= i < n :: |I[ch][i]| == n
  }

  /** Pixel (i, j) of channel ch, or 0 outside the image. */
  ghost function Pixel(I: seq<seq<seq<real>>>, c: nat, n: nat, ch: int, i: int, j: int): real
    requires IsImage(I, c, n)
  {
    if 0 <= ch < c && 0 <= i < n && 0 <= j < n then I[ch][i][j] else 0.0
  }

  /** The weight K[o][ch][a][b], or 0 outside the layer's filter. */
  ghost function Weight(L: ConvLayer, o: int, ch: int, a: int, b: int): real
    requires ValidConv(L)
  {
    if 0 <= o < L.cOut && 0 <= ch < L.cIn && 0 <= a < L.k && 0 <= b < L.k
    then L.K[o][ch][a][b] else 0.0
  }

  /** ZeroPadding2D(p): each channel of I with p zeros on each side. */
  ghost opaque function ZeroPad(I: seq<seq<seq<real>>>, c: nat, n: nat, p: nat): (P: seq<seq<seq<real>>>)
    requires IsImage(I, c, n)
    ensures IsImage(P, c, n + 2 * p)
  {
    seq(c, ch requires 0 <= ch < c =>
      seq(n + 2 * p, i requires 0 <= i < n + 2 * p =>
        seq(n + 2 * p, j requires 0 <= j < n + 2 * p => Pixel(I, c, n, ch, i - p, j - p))))
  }

  /**
    * The layer's output before ReLU, as Keras computes it: ZeroPadding2D(pad),
    * then Conv2D(cOut, k, strides=stride, padding='valid', use_bias=False).
    * Output channel o at row y and column x is the sum over input channels c and
    * kernel positions (a, b) of K[o][c][a][b] * P[c][y*stride + a][x*stride + b],
    * where P is the padded input and K[o][c][a][b] is Keras' kernel[a][b][c][o].
    * Weight and Pixel are 0 outside the filter and the image only so that every
    * term is defined.
    */
  ghost opaque function Conv2DZeroPad(L: ConvLayer, I: seq<seq<seq<real>>>): (Y: seq<seq<seq<real>>>)
    requires ValidConv(L) && IsImage(I, L.cIn, L.n)
    ensures IsImage(Y, L.cOut, OutSize(L))
  {
    var P := ZeroPad(I, L.cIn, L.n, L.pad);
    var m := OutSize(L);
    seq(L.cOut, o requires 0 <= o < L.cOut =>
      seq(m, y requires 0 <= y < m =>
        seq(m, x requires 0 <= x < m =>
          Sum3(L.cIn, L.k, L.k, (c, a, b) =>
                 Weight(L, o, c, a, b) *
                 Pixel(P, L.cIn, L.n + 2 * L.pad, c, y * L.stride + a, x * L.stride + b)))))
  }

  /**
    * The image as a vector, channel by channel, then row by row, as the
    * certifier orders a convolutional layer's inputs and outputs: pixel (i, j)
    * of channel ch is entry ch*n*n + i*n + j (make_certifier_format.py).
    */
  ghost opaque function Vec(I: seq<seq<seq<real>>>, c: nat, n: nat): (v: Vector)
    requires IsImage(I, c, n) && c > 0 && n > 0
    ensures |v| == c * (n * n)
  {
    MulPos(n, n);
    MulPos(c, n * n);
    seq(c * (n * n), t requires 0 <= t < c * (n * n) =>
      DivBound(t, c, n * n);
      I[t / (n * n)][(t / n) % n][t % n])
  }

  /** The vector v as an image: the inverse of Vec. */
  ghost opaque function Unvec(v: seq<real>, c: nat, n: nat): (I: seq<seq<seq<real>>>)
    requires |v| == c * (n * n)
    ensures IsImage(I, c, n)
  {
    seq(c, ch requires 0 <= ch < c =>
      seq(n, i requires 0 <= i < n =>
        seq(n, j requires 0 <= j < n =>
          IndexBound(ch, i, j, c, n);
          v[ch * (n * n) + i * n + j])))
  }

  /**
    * The link between the certifier's ghost network and the CNN: the layer's
    * banded doubly block Toeplitz matrix times the input image, as a vector, is
    * the output of Conv2DZeroPad, as a vector.
    */
  lemma ToeplitzIsConv(L: ConvLayer, I: seq<seq<seq<real>>>)
    requires ValidConv(L) && IsImage(I, L.cIn, L.n)
    ensures MV(ToeplitzMatrix(L), Vec(I, L.cIn, L.n)) ==
            Vec(Conv2DZeroPad(L, I), L.cOut, OutSize(L))
  {
    var lhs := MV(ToeplitzMatrix(L), Vec(I, L.cIn, L.n));
    var rhs := Vec(Conv2DZeroPad(L, I), L.cOut, OutSize(L));
    forall r | 0 <= r < |lhs|
      ensures lhs[r] == rhs[r]
    {
      ToeplitzRowIsConv(L, I, r);
    }
  }

  /** The same for every input vector v, as the certifier's NN applies the layer. */
  lemma ToeplitzIsConvVec(L: ConvLayer, v: Vector)
    requires ValidConv(L) && |v| == L.cIn * (L.n * L.n)
    ensures MV(ToeplitzMatrix(L), v) ==
            Vec(Conv2DZeroPad(L, Unvec(v, L.cIn, L.n)), L.cOut, OutSize(L))
  {
    VecUnvec(v, L.cIn, L.n);
    ToeplitzIsConv(L, Unvec(v, L.cIn, L.n));
  }

  /* ------------------------------- The proof -------------------------------- */

  /** Row r's terms: (c, i, j) => K[o][c][i + dy][j + dx] * I[c][i][j]. */
  ghost function MatrixTerms(L: ConvLayer, I: seq<seq<seq<real>>>, o: int, dy: int, dx: int):
    (int, int, int) -> real
    requires ValidConv(L) && IsImage(I, L.cIn, L.n)
  {
    (c, i, j) => Weight(L, o, c, i + dy, j + dx) * Pixel(I, L.cIn, L.n, c, i, j)
  }

  /** The convolution's terms: (c, a, b) => K[o][c][a][b] * I[c][a - dy][b - dx]. */
  ghost function ConvTerms(L: ConvLayer, I: seq<seq<seq<real>>>, o: int, dy: int, dx: int):
    (int, int, int) -> real
    requires ValidConv(L) && IsImage(I, L.cIn, L.n)
  {
    (c, a, b) => Weight(L, o, c, a, b) * Pixel(I, L.cIn, L.n, c, a - dy, b - dx)
  }

  /**
    * Output r of ToeplitzIsConv, at channel o, row y and column x. Row r of the
    * matrix sums over input pixels (i, j), the convolution over kernel
    * positions (a, b) = (i + dy, j + dx).
    */
  lemma ToeplitzRowIsConv(L: ConvLayer, I: seq<seq<seq<real>>>, r: int)
    requires ValidConv(L) && IsImage(I, L.cIn, L.n)
    requires 0 <= r < L.cOut * (OutSize(L) * OutSize(L))
    ensures Dot(ToeplitzMatrix(L)[r], Vec(I, L.cIn, L.n)) ==
            Vec(Conv2DZeroPad(L, I), L.cOut, OutSize(L))[r]
  {
    var m := OutSize(L);
    MulPos(m, m);
    DivBound(r, L.cOut, m * m);
    var o, y, x := r / (m * m), (r / m) % m, r % m;
    var dy, dx := L.pad - y * L.stride, L.pad - x * L.stride;
    MatrixRowSum(L, I, r, o, y, x);
    ReindexTerms(L, I, o, dy, dx);
    Sum3Reindex(L.cIn, L.n, L.k, dy, dx, MatrixTerms(L, I, o, dy, dx), ConvTerms(L, I, o, dy, dx));
    ConvRowSum(L, I, r, o, y, x);
  }

  /** Row r of the matrix times the input, as a sum over channels and pixels. */
  lemma MatrixRowSum(L: ConvLayer, I: seq<seq<seq<real>>>, r: int, o: int, y: int, x: int)
    requires ValidConv(L) && IsImage(I, L.cIn, L.n)
    requires 0 <= r < L.cOut * (OutSize(L) * OutSize(L))
    requires o == r / (OutSize(L) * OutSize(L)) && y == (r / OutSize(L)) % OutSize(L) &&
             x == r % OutSize(L) && 0 <= o < L.cOut
    ensures Dot(ToeplitzMatrix(L)[r], Vec(I, L.cIn, L.n)) ==
            Sum3(L.cIn, L.n, L.n, MatrixTerms(L, I, o, L.pad - y * L.stride, L.pad - x * L.stride))
  {
    var n, C := L.n, L.cIn;
    var row, v := ToeplitzMatrix(L)[r], Vec(I, C, n);
    var h := MatrixTerms(L, I, o, L.pad - y * L.stride, L.pad - x * L.stride);
    DotSumRange(row, v);
    var F := t => if 0 <= t < |row| then row[t] * v[t] else 0.0;
    forall c, i, j | 0 <= c < C && 0 <= i < n && 0 <= j < n
      ensures F(c * (n * n) + i * n + j) == h(c, i, j)
    {
      MatrixTerm(L, I, r, o, y, x, c, i, j);
    }
    SumRangeVec(C, n, F, h);
  }

  /** Term c*n*n + i*n + j of row r of the matrix times the input. */
  lemma MatrixTerm(L: ConvLayer, I: seq<seq<seq<real>>>, r: int, o: int, y: int, x: int,
                   c: int, i: int, j: int)
    requires ValidConv(L) && IsImage(I, L.cIn, L.n)
    requires 0 <= r < L.cOut * (OutSize(L) * OutSize(L))
    requires o == r / (OutSize(L) * OutSize(L)) && y == (r / OutSize(L)) % OutSize(L) &&
             x == r % OutSize(L) && 0 <= o < L.cOut
    requires 0 <= c < L.cIn && 0 <= i < L.n && 0 <= j < L.n
    ensures 0 <= c * (L.n * L.n) + i * L.n + j < L.cIn * (L.n * L.n)
    ensures ToeplitzMatrix(L)[r][c * (L.n * L.n) + i * L.n + j] *
            Vec(I, L.cIn, L.n)[c * (L.n * L.n) + i * L.n + j] ==
            MatrixTerms(L, I, o, L.pad - y * L.stride, L.pad - x * L.stride)(c, i, j)
  {
    var n := L.n;
    var t := c * (n * n) + i * n + j;
    VecIndex(c, i, j, n);
    IndexBound(c, i, j, L.cIn, n);
    VecAt(I, L.cIn, n, t);
    assert ToeplitzMatrix(L)[r][t] == ToeplitzEntry(L, o, y, x, c, i, j);
  }

  /** Output r of Conv2DZeroPad, as the sum of the convolution's terms. */
  lemma ConvRowSum(L: ConvLayer, I: seq<seq<seq<real>>>, r: int, o: int, y: int, x: int)
    requires ValidConv(L) && IsImage(I, L.cIn, L.n)
    requires 0 <= r < L.cOut * (OutSize(L) * OutSize(L))
    requires o == r / (OutSize(L) * OutSize(L)) && y == (r / OutSize(L)) % OutSize(L) &&
             x == r % OutSize(L) && 0 <= o < L.cOut
    ensures Vec(Conv2DZeroPad(L, I), L.cOut, OutSize(L))[r] ==
            Sum3(L.cIn, L.k, L.k, ConvTerms(L, I, o, L.pad - y * L.stride, L.pad - x * L.stride))
  {
    var m := OutSize(L);
    VecAt(Conv2DZeroPad(L, I), L.cOut, m, r);
    ConvAt(L, I, o, y, x);
  }

  /** Output (o, y, x) of Conv2DZeroPad, with the padding moved into Pixel's index. */
  lemma ConvAt(L: ConvLayer, I: seq<seq<seq<real>>>, o: int, y: int, x: int)
    requires ValidConv(L) && IsImage(I, L.cIn, L.n)
    requires 0 <= o < L.cOut && 0 <= y < OutSize(L) && 0 <= x < OutSize(L)
    ensures Conv2DZeroPad(L, I)[o][y][x] ==
            Sum3(L.cIn, L.k, L.k, ConvTerms(L, I, o, L.pad - y * L.stride, L.pad - x * L.stride))
  {
    reveal Conv2DZeroPad();
    var P := ZeroPad(I, L.cIn, L.n, L.pad);
    var hP := (c, a, b) =>
        Weight(L, o, c, a, b) *
        Pixel(P, L.cIn, L.n + 2 * L.pad, c, y * L.stride + a, x * L.stride + b);
    var hK := ConvTerms(L, I, o, L.pad - y * L.stride, L.pad - x * L.stride);
    assert Conv2DZeroPad(L, I)[o][y][x] == Sum3(L.cIn, L.k, L.k, hP);
    forall c, a, b | 0 <= c < L.cIn && 0 <= a < L.k && 0 <= b < L.k
      ensures hP(c, a, b) == hK(c, a, b)
    {
      PixelZeroPad(I, L.cIn, L.n, L.pad, c, y * L.stride + a, x * L.stride + b);
    }
    Sum3Ext(L.cIn, L.k, L.k, hP, hK);
  }

  /**
    * The conditions of Sum3Reindex for row r: the matrix's terms are the
    * convolution's terms at (a, b) = (i + dy, j + dx), and the convolution's
    * terms are zero outside the filter or the image.
    */
  lemma ReindexTerms(L: ConvLayer, I: seq<seq<seq<real>>>, o: int, dy: int, dx: int)
    requires ValidConv(L) && IsImage(I, L.cIn, L.n)
    ensures forall c, i, j | 0 <= c < L.cIn && 0 <= i < L.n && 0 <= j < L.n ::
              MatrixTerms(L, I, o, dy, dx)(c, i, j) == ConvTerms(L, I, o, dy, dx)(c, i + dy, j + dx)
    ensures forall c, a, b | 0 <= c < L.cIn &&
                             !(0 <= a < L.k && 0 <= b < L.k && dy <= a < L.n + dy && dx <= b < L.n + dx) ::
              ConvTerms(L, I, o, dy, dx)(c, a, b) == 0.0
  {
    var hT, hK := MatrixTerms(L, I, o, dy, dx), ConvTerms(L, I, o, dy, dx);
    forall c, i, j | 0 <= c < L.cIn && 0 <= i < L.n && 0 <= j < L.n
      ensures hT(c, i, j) == hK(c, i + dy, j + dx)
    {
      assert (i + dy) - dy == i && (j + dx) - dx == j;
    }
    forall c, a, b | 0 <= c < L.cIn &&
                     !(0 <= a < L.k && 0 <= b < L.k && dy <= a < L.n + dy && dx <= b < L.n + dx)
      ensures hK(c, a, b) == 0.0
    {
      if !(0 <= a < L.k && 0 <= b < L.k) {
        assert Weight(L, o, c, a, b) == 0.0;
      } else {
        assert Pixel(I, L.cIn, L.n, c, a - dy, b - dx) == 0.0;
      }
    }
  }

  lemma PixelZeroPad(I: seq<seq<seq<real>>>, c: nat, n: nat, p: nat, ch: int, i: int, j: int)
    requires IsImage(I, c, n)
    ensures Pixel(ZeroPad(I, c, n, p), c, n + 2 * p, ch, i, j) == Pixel(I, c, n, ch, i - p, j - p)
  {
    reveal ZeroPad();
  }

  /** Entry t of Vec(I). */
  lemma VecAt(I: seq<seq<seq<real>>>, c: nat, n: nat, t: int)
    requires IsImage(I, c, n) && c > 0 && n > 0 && 0 <= t < c * (n * n)
    ensures 0 <= t / (n * n) < c && 0 <= (t / n) % n < n && 0 <= t % n < n
    ensures Vec(I, c, n)[t] == I[t / (n * n)][(t / n) % n][t % n]
  {
    reveal Vec();
    MulPos(n, n);
    DivBound(t, c, n * n);
  }

  /** Pixel (i, j) of channel ch of Unvec(v). */
  lemma UnvecAt(v: seq<real>, c: nat, n: nat, ch: int, i: int, j: int)
    requires |v| == c * (n * n) && 0 <= ch < c && 0 <= i < n && 0 <= j < n
    ensures 0 <= ch * (n * n) + i * n + j < |v|
    ensures Unvec(v, c, n)[ch][i][j] == v[ch * (n * n) + i * n + j]
  {
    reveal Unvec();
    IndexBound(ch, i, j, c, n);
  }

  lemma VecUnvec(v: Vector, c: nat, n: nat)
    requires |v| == c * (n * n) && c > 0 && n > 0
    ensures Vec(Unvec(v, c, n), c, n) == v
  {
    var U := Unvec(v, c, n);
    var w := Vec(U, c, n);
    forall t | 0 <= t < |v|
      ensures w[t] == v[t]
    {
      VecAt(U, c, n, t);
      UnvecAt(v, c, n, t / (n * n), (t / n) % n, t % n);
      VecIndexInverse(t, n);
    }
  }

  /* ---------------------------------- Sums ---------------------------------- */

  /** f(lo) + f(lo + 1) + ... + f(hi - 1), or 0 if hi <= lo. */
  ghost function SumRange(lo: int, hi: int, f: int -> real): real
    decreases hi - lo
  {
    if lo >= hi then 0.0 else f(lo) + SumRange(lo + 1, hi, f)
  }

  /** The sum of h(c, i, j) over c < C, i < n1 and j < n2. */
  ghost opaque function Sum3(C: int, n1: int, n2: int, h: (int, int, int) -> real): real {
    SumRange(0, C, c => SumRange(0, n1, i => SumRange(0, n2, j => h(c, i, j))))
  }

  function MaxI(a: int, b: int): int { if a < b then b else a }

  function MinI(a: int, b: int): int { if a < b then a else b }

  /** Dot(u, v) as the sum of u[t] * v[t] over t. */
  lemma DotSumRange(u: Vector, v: Vector)
    requires |u| == |v|
    ensures Dot(u, v) == SumRange(0, |u|, t => if 0 <= t < |u| then u[t] * v[t] else 0.0)
    decreases |u|
  {
    reveal Dot();
    if |u| > 1 {
      var u', v' := u[1..], v[1..];
      DotSumRange(u', v');
      SumRangeShift(0, |u'|, 1, t => if 0 <= t < |u'| then u'[t] * v'[t] else 0.0,
                    t => if 0 <= t < |u| then u[t] * v[t] else 0.0);
    }
  }

  lemma SumRangeExt(lo: int, hi: int, f: int -> real, g: int -> real)
    requires forall t | lo <= t < hi :: f(t) == g(t)
    ensures SumRange(lo, hi, f) == SumRange(lo, hi, g)
    decreases hi - lo
  {
    if lo < hi { SumRangeExt(lo + 1, hi, f, g); }
  }

  lemma SumRangeZero(lo: int, hi: int, f: int -> real)
    requires forall t | lo <= t < hi :: f(t) == 0.0
    ensures SumRange(lo, hi, f) == 0.0
    decreases hi - lo
  {
    if lo < hi { SumRangeZero(lo + 1, hi, f); }
  }

  lemma SumRangeSplit(lo: int, mid: int, hi: int, f: int -> real)
    requires lo <= mid <= hi
    ensures SumRange(lo, hi, f) == SumRange(lo, mid, f) + SumRange(mid, hi, f)
    decreases mid - lo
  {
    if lo < mid { SumRangeSplit(lo + 1, mid, hi, f); }
  }

  /** Shifting the index by d. */
  lemma SumRangeShift(lo: int, hi: int, d: int, f: int -> real, g: int -> real)
    requires forall t | lo <= t < hi :: f(t) == g(t + d)
    ensures SumRange(lo, hi, f) == SumRange(lo + d, hi + d, g)
    decreases hi - lo
  {
    if lo < hi { SumRangeShift(lo + 1, hi, d, f, g); }
  }

  /** Where f is zero outside [A, B), only [A, B) counts. */
  lemma SumRangeSupport(lo: int, hi: int, A: int, B: int, f: int -> real)
    requires forall t | lo <= t < hi && (t < A || t >= B) :: f(t) == 0.0
    ensures SumRange(lo, hi, f) == SumRange(MaxI(lo, A), MinI(hi, B), f)
  {
    var a, b := MaxI(lo, A), MinI(hi, B);
    if a >= b {
      SumRangeZero(lo, hi, f);
    } else {
      SumRangeSplit(lo, a, hi, f);
      SumRangeSplit(a, b, hi, f);
      SumRangeZero(lo, a, f);
      SumRangeZero(b, hi, f);
    }
  }

  /**
    * Reindexing t = j + d: the sum of g(j) over j < n equals the sum of f(t) over
    * t < k when g(j) == f(j + d) and f(t) is zero unless 0 <= t < k and
    * d <= t < n + d.
    */
  lemma SumRangeReindex(n: int, k: int, d: int, f: int -> real, g: int -> real)
    requires forall j | 0 <= j < n :: g(j) == f(j + d)
    requires forall t | t < 0 || t >= k || t < d || t >= n + d :: f(t) == 0.0
    ensures SumRange(0, n, g) == SumRange(0, k, f)
  {
    SumRangeShift(0, n, d, g, f);
    SumRangeSupport(d, n + d, 0, k, f);
    SumRangeSupport(0, k, d, n + d, f);
    assert MaxI(d, 0) == MaxI(0, d) && MinI(n + d, k) == MinI(k, n + d);
  }

  /** A sum over [0, A*B) as A blocks of B. */
  lemma SumRangeBlocks(A: nat, B: nat, f: int -> real, g: int -> real)
    requires forall q | 0 <= q < A :: g(q) == SumRange(q * B, q * B + B, f)
    ensures SumRange(0, A * B, f) == SumRange(0, A, g)
    decreases A
  {
    if A > 0 {
      SumRangeBlocks(A - 1, B, f, g);
      MulMono(0, A - 1, B);
      assert (A - 1) * B + B == A * B;
      SumRangeSplit(0, (A - 1) * B, A * B, f);
      SumRangeSplit(0, A - 1, A, g);
    }
  }

  /** A sum over t < C*n*n as a sum over c, i and j, with t = c*n*n + i*n + j. */
  lemma SumRangeVec(C: nat, n: nat, F: int -> real, h: (int, int, int) -> real)
    requires n > 0
    requires forall c, i, j | 0 <= c < C && 0 <= i < n && 0 <= j < n ::
               F(c * (n * n) + i * n + j) == h(c, i, j)
    ensures SumRange(0, C * (n * n), F) == Sum3(C, n, n, h)
  {
    reveal Sum3();
    var nn := n * n;
    var G := c => SumRange(c * nn, c * nn + nn, F);
    var H := c => SumRange(0, n, i => SumRange(0, n, j => h(c, i, j)));
    SumRangeBlocks(C, nn, F, G);
    forall c | 0 <= c < C
      ensures G(c) == H(c)
    {
      var Fc := u => F(c * nn + u);
      SumRangeShift(0, nn, c * nn, Fc, F);
      var Gc := i => SumRange(i * n, i * n + n, Fc);
      var Hc := i => SumRange(0, n, j => h(c, i, j));
      SumRangeBlocks(n, n, Fc, Gc);
      forall i | 0 <= i < n
        ensures Gc(i) == Hc(i)
      {
        var Fci := j => Fc(i * n + j);
        SumRangeShift(0, n, i * n, Fci, Fc);
        forall j | 0 <= j < n
          ensures Fci(j) == h(c, i, j)
        {
          assert c * nn + (i * n + j) == c * (n * n) + i * n + j;
        }
        SumRangeExt(0, n, Fci, j => h(c, i, j));
      }
      SumRangeExt(0, n, Gc, Hc);
    }
    SumRangeExt(0, C, G, H);
  }

  /**
    * Reindexing (a, b) = (i + dy, j + dx) in each channel c: the sum of
    * g(c, i, j) over i, j < n equals the sum of f(c, a, b) over a, b < k when
    * g(c, i, j) == f(c, i + dy, j + dx) and f(c, a, b) is zero unless a, b < k
    * and a - dy, b - dx < n.
    */
  lemma Sum3Reindex(C: int, n: int, k: int, dy: int, dx: int,
                    g: (int, int, int) -> real, f: (int, int, int) -> real)
    requires forall c, i, j | 0 <= c < C && 0 <= i < n && 0 <= j < n ::
               g(c, i, j) == f(c, i + dy, j + dx)
    requires forall c, a, b | 0 <= c < C &&
                              !(0 <= a < k && 0 <= b < k && dy <= a < n + dy && dx <= b < n + dx) ::
               f(c, a, b) == 0.0
    ensures Sum3(C, n, n, g) == Sum3(C, k, k, f)
  {
    reveal Sum3();
    var G := c => SumRange(0, n, i => SumRange(0, n, j => g(c, i, j)));
    var F := c => SumRange(0, k, i => SumRange(0, k, j => f(c, i, j)));
    forall c | 0 <= c < C
      ensures G(c) == F(c)
    {
      var rowG := i => SumRange(0, n, j => g(c, i, j));
      var rowF := i => SumRange(0, k, j => f(c, i, j));
      forall i | 0 <= i < n
        ensures rowG(i) == rowF(i + dy)
      {
        SumRangeReindex(n, k, dx, j => f(c, i + dy, j), j => g(c, i, j));
      }
      forall a | a < 0 || a >= k || a < dy || a >= n + dy
        ensures rowF(a) == 0.0
      {
        SumRangeZero(0, k, j => f(c, a, j));
      }
      SumRangeReindex(n, k, dy, rowF, rowG);
    }
    SumRangeExt(0, C, G, F);
  }

  lemma Sum3Ext(C: int, n1: int, n2: int, g: (int, int, int) -> real, f: (int, int, int) -> real)
    requires forall c, i, j | 0 <= c < C && 0 <= i < n1 && 0 <= j < n2 :: g(c, i, j) == f(c, i, j)
    ensures Sum3(C, n1, n2, g) == Sum3(C, n1, n2, f)
  {
    reveal Sum3();
    var G := c => SumRange(0, n1, i => SumRange(0, n2, j => g(c, i, j)));
    var F := c => SumRange(0, n1, i => SumRange(0, n2, j => f(c, i, j)));
    forall c | 0 <= c < C
      ensures G(c) == F(c)
    {
      var rowG := i => SumRange(0, n2, j => g(c, i, j));
      var rowF := i => SumRange(0, n2, j => f(c, i, j));
      forall i | 0 <= i < n1
        ensures rowG(i) == rowF(i)
      {
        SumRangeExt(0, n2, j => g(c, i, j), j => f(c, i, j));
      }
      SumRangeExt(0, n1, rowG, rowF);
    }
    SumRangeExt(0, C, G, F);
  }

  /* ======================= Integer facts for indexing ======================= */

  lemma MulPos(a: int, b: int)
    requires a > 0 && b > 0
    ensures a * b > 0
  {}

  lemma DivBound(r: int, A: int, B: int)
    requires 0 <= r < A * B && B > 0
    ensures 0 <= r / B < A
  {
    if r / B >= A {
      MulMono(A, r / B, B);
      assert false;
    }
  }

  lemma MulMono(a: int, b: int, c: int)
    requires a <= b && c >= 0
    ensures a * c <= b * c
  {}

  /** For 0 <= r < B: (q*B + r) / B == q and (q*B + r) % B == r. */
  lemma DivModUnique(q: int, r: int, B: int)
    requires B > 0 && 0 <= r < B
    ensures (q * B + r) / B == q && (q * B + r) % B == r
  {
    var t := q * B + r;
    var q', r' := t / B, t % B;
    assert t == q' * B + r' && 0 <= r' < B;
    if q' < q {
      MulMono(q' + 1, q, B);
    } else if q' > q {
      MulMono(q + 1, q', B);
    }
  }

  /** Entry ch*n*n + i*n + j of a vectorised image is pixel (i, j) of channel ch. */
  lemma VecIndex(ch: int, i: int, j: int, n: int)
    requires 0 <= ch && 0 <= i < n && 0 <= j < n
    ensures (ch * (n * n) + i * n + j) / (n * n) == ch
    ensures ((ch * (n * n) + i * n + j) / n) % n == i
    ensures (ch * (n * n) + i * n + j) % n == j
  {
    assert ch * (n * n) + i * n + j == (ch * n + i) * n + j;
    DivModUnique(ch * n + i, j, n);
    DivModUnique(ch, i, n);
    MulMono(i, n - 1, n);
    MulMono(0, i, n);
    assert ch * (n * n) + i * n + j == ch * (n * n) + (i * n + j);
    DivModUnique(ch, i * n + j, n * n);
  }

  /** Every t is ch*n*n + i*n + j for ch = t / (n*n), i = (t / n) % n and j = t % n. */
  lemma VecIndexInverse(t: int, n: int)
    requires 0 <= t && n > 0
    ensures t == (t / (n * n)) * (n * n) + ((t / n) % n) * n + t % n
  {
    var q, r := t / n, t % n;
    var q2, r2 := q / n, q % n;
    assert t == q * n + r;
    assert q == q2 * n + r2;
    MulMono(r2, n - 1, n);
    MulMono(0, r2, n);
    assert t == q2 * (n * n) + (r2 * n + r);
    DivModUnique(q2, r2 * n + r, n * n);
  }

  lemma IndexBound(ch: int, i: int, j: int, c: int, n: int)
    requires 0 <= ch < c && 0 <= i < n && 0 <= j < n
    ensures 0 <= ch * (n * n) + i * n + j < c * (n * n)
  {
    MulMono(i, n - 1, n);
    MulMono(ch, c - 1, n * n);
    MulMono(0, ch, n * n);
    MulMono(0, i, n);
  }
}
