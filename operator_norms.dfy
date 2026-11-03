include "basic_arithmetic.dfy"
include "linear_algebra.dfy"
include "convolution.dfy"

module OperatorNorms {
  import opened BasicArithmetic
  import opened LinearAlgebra
  import opened Convolution

  /* ============================ Ghost Functions ============================= */

  ghost function FrobeniusNorm(m: Matrix): real {
    Sqrt(SumPositiveMatrix(SquareMatrixElements(m)))
  }

  ghost opaque function {:axiom} SpecNorm(m: Matrix): (r: real)
    ensures r >= 0.0
    ensures IsSpecNormUpperBound(r, m)
    ensures !exists x: real :: 0.0 <= x < r && IsSpecNormUpperBound(x, m)

  ghost predicate IsSpecNormUpperBound(s: real, m: Matrix) {
    s >= 0.0 && forall v: Vector | |v| == |m[0]| :: L2(MV(m, v)) <= s * L2(v)
  }

  /* ================================ Methods ================================= */

  /** Generates the spectral norm for each matrix in n. */
  method GenerateSpecNorms(n: seq<Matrix>, GRAM_ITERATIONS: int) returns (r: seq<real>)
    requires GRAM_ITERATIONS >= 0
    ensures |r| == |n|
    ensures forall i | 0 <= i < |n| :: IsSpecNormUpperBound(r[i], n[i])
  {
    var i := 0;
    r := [];
    while i < |n|
      invariant 0 <= i == |r| <= |n|
      invariant forall j | 0 <= j < i :: IsSpecNormUpperBound(r[j], n[j])
    {
      if DEBUG { print "{ \"debug_msg\": \"Generating spectral norm ", i, " of ", |n|, "...\" },\n"; }
      var specNorm := GramIterationSimple(n[i], GRAM_ITERATIONS);
      assert specNorm >= SpecNorm(n[i]);
      r := r + [specNorm];
      i := i + 1;
    }
  }

  /**
    * Generates a spectral norm upper bound for each layer of a CNN: Gram iteration
    * for a dense layer, as above, and Algorithm 3 (norm2_toep) of Delattre et al.
    * (2024) for a convolutional layer.
    */
  method GenerateCNNSpecNorms(ls: seq<CNNLayer>, GRAM_ITERATIONS: int, CONV_ITERATIONS: nat)
    returns (r: seq<real>)
    requires GRAM_ITERATIONS >= 0 && CONV_ITERATIONS >= 1
    requires forall i | 0 <= i < |ls| :: ValidLayer(ls[i])
    ensures |r| == |ls|
    ensures forall i | 0 <= i < |ls| :: IsSpecNormUpperBound(r[i], LayerMatrix(ls[i]))
  {
    ghost var n := seq(|ls|, j requires 0 <= j < |ls| => LayerMatrix(ls[j]));
    var i := 0;
    r := [];
    while i < |ls|
      invariant 0 <= i == |r| <= |ls|
      invariant forall j | 0 <= j < i :: IsSpecNormUpperBound(r[j], n[j])
    {
      if DEBUG { print "{ \"debug_msg\": \"Generating spectral norm ", i, " of ", |ls|, "...\" },\n"; }
      var specNorm := LayerSpecNorm(ls[i], GRAM_ITERATIONS, CONV_ITERATIONS);
      r := r + [specNorm];
      i := i + 1;
    }
  }

  /** The bound for one layer of a CNN. */
  method LayerSpecNorm(l: CNNLayer, GRAM_ITERATIONS: int, CONV_ITERATIONS: nat) returns (s: real)
    requires GRAM_ITERATIONS >= 0 && CONV_ITERATIONS >= 1 && ValidLayer(l)
    ensures IsSpecNormUpperBound(s, LayerMatrix(l))
  {
    if l.Dense? {
      s := GramIterationSimple(l.W, GRAM_ITERATIONS);
    } else {
      s := Norm2Toep(l.C, CONV_ITERATIONS);
    }
  }

  method FrobeniusNormUpperBound(m: Matrix) returns (r: real)
    ensures r >= FrobeniusNorm(m)
  {
    var n := SquareMatrixElementsImpl(m);
    var x := SumPositiveMatrixImpl(n);
    r := SqrtUpperBound(x);
  }

  // method FrobeniusNormUpperBound(m: Matrix) returns (r: real)
  //   ensures r >= FrobeniusNorm(m)
  // {
  //   if DEBUG { print "Computing frobenius norm upper bound for matrix of size ", |m|, "x", |m[0]|, "\n"; }
  //   r := 0.0;
  //   for i := 0 to |m|
  //     invariant r >= 0.0
  //     invariant i != 0 ==> r == SumPositiveMatrix(SquareMatrixElements(m[..i]))
  //   {
  //     for j := 0 to |m[i]|
  //       invariant r >= 0.0

  //     {
  //       r := r + m[i][j] * m[i][j];
  //     }
  //   }
  //   r := SqrtUpperBound(r);
  // }


  lemma SumPositiveIsZero(v: Vector)
    requires forall i | 0 <= i < |v| :: 0.0 <= v[i]
    requires SumPositive(v) == 0.0
    ensures forall i | 0 <= i < |v| :: v[i] == 0.0
  {}

  lemma SumPositiveMatrixIsZero(m: Matrix)
    requires forall i,j | 0 <= i < |m| && 0 <= j < |m[0]| :: 0.0 <= m[i][j]
    requires SumPositiveMatrix(m) == 0.0
    ensures forall i,j | 0 <= i < |m| && 0 <= j < |m[0]| :: m[i][j] == 0.0
  {
    SumPositiveIsZero(m[0]);
  }

  lemma MultIsZero(x: real, y: real)
    requires x * y == 0.0
    ensures x == 0.0 || y == 0.0
  {}

  lemma MultNonZero(x: real, y: real)
    requires x * y != 0.0
    ensures x != 0.0 && y != 0.0
  {}

  lemma SquareIsZero(x: real)
    requires Square(x) == 0.0
    ensures x == 0.0
  {
    MultIsZero(x,x);
  }

  lemma ApplySquareIsZero(v: Vector)
    requires forall i | 0 <= i < |v| :: v[i] * v[i] == 0.0
    ensures  forall i | 0 <= i < |v| :: v[i] == 0.0
  {
    reveal Apply();
    if |v| == 1 {
      SquareIsZero(v[0]);
    } else {
      SquareIsZero(v[0]);
      ApplySquareIsZero(v[1..]);
    }
  }

  lemma SquareMatrixElementsSize(m: Matrix, r: Matrix)
    requires r == SquareMatrixElements(m)
    ensures Rows(r) == Rows(m) && Cols(r) == Cols(m)
  {
    if |m| == 1 {
    } else {
      SquareMatrixElementsSize(m[1..],r[1..]);
    }
  }
  lemma SquareMatrixElementsElements(m: Matrix, r: Matrix, i: nat, j: nat)
    requires r == SquareMatrixElements(m)
    requires i <= i < |r| && 0 <= j < |r[0]|
    ensures Rows(r) == Rows(m) && Cols(r) == Cols(m)
    ensures r[i][j] == m[i][j] * m[i][j]
  {
    if |m| == 1 {
    } else {
      if i == 0 {
        SquareMatrixElementsElements(m[1..],r[1..],0,j);
      } else {
        SquareMatrixElementsElements(m[1..],r[1..],i-1,j);
      }
    }
  }

  lemma SquareMatrixElementsIsZero(m: Matrix, r: Matrix)
    requires r == SquareMatrixElements(m)
    requires forall i,j | 0 <= i < |r| && 0 <= j < |r[0]| :: r[i][j] == 0.0
    ensures forall i,j | 0 <= i < |m| && 0 <= j < |m[0]| :: m[i][j] == 0.0
  {
    SquareMatrixElementsSize(m,r);
    forall i:nat,j:nat | 0 <= i < |m| && 0 <= j < |m[0]| ensures m[i][j] == 0.0 {
      SquareMatrixElementsElements(m,r,i,j);
      SquareIsZero(m[i][j]);
    }
  }

  function VectorIsZero(v: Vector) : (b: bool)
    ensures b <==> forall i | 0 <= i < |v| :: v[i] == 0.0
  {
    if |v| == 1 then v[0] == 0.0 else v[0] == 0.0 && VectorIsZero(v[1..])
  }

  function MatrixIsZero(m: Matrix) : (b: bool)
    ensures b <==> forall i,j | 0 <= i < |m| && 0 <= j < |m[0]| :: m[i][j] == 0.0
  {
    if |m| == 1 then VectorIsZero(m[0]) else VectorIsZero(m[0]) && MatrixIsZero(m[1..])
  }

  lemma SqrtIsZero(x: real)
    requires x >= 0.0
    requires Sqrt(x) == 0.0
    ensures x == 0.0
  {}

  lemma FrobeniusNormIsZero(m: Matrix)
    requires FrobeniusNorm(m) == 0.0
    ensures MatrixIsZero(m)
  {
    SqrtIsZero(SumPositiveMatrix(SquareMatrixElements(m)));
    SumPositiveMatrixIsZero(SquareMatrixElements(m));
    SquareMatrixElementsIsZero(m,SquareMatrixElements(m));
  }

  lemma SafeFrobeniusNorm(m: Matrix)
    requires !MatrixIsZero(m)
    ensures FrobeniusNorm(m) != 0.0
  {
    forall n:Matrix | FrobeniusNorm(n) == 0.0 ensures MatrixIsZero(n) { FrobeniusNormIsZero(n); }
  }


  function MatrixSymmetric(m: Matrix) : (b: bool)
    requires Rows(m) == Cols(m)
  {
    forall i,j | 0 <= i < Rows(m) && 0 <= j < i :: m[i][j] == m[j][i]
  }

  method Truncate(m: Matrix) returns (r: Matrix, e: Matrix)
    requires Rows(m) == Cols(m)
    requires MatrixSymmetric(m)
    ensures SpecNorm(m) <= SpecNorm(r) + SpecNorm(e)
  {
    opaque ensures Abs(SpecNorm(r)-SpecNorm(m)) <= SpecNorm(e) {
      var rM := new real[Rows(m), Rows(m)];
      var eM := new real[Rows(m), Rows(m)];
      var i := 0;
      while i < rM.Length0
        invariant 0 <= i <= Rows(m)
        invariant forall x,y | 0 <= x < i && 0 <= y < Cols(m) :: rM[x,y] == m[x][y] + eM[x,y]
        invariant forall x,y | 0 <= x < i && 0 <= y < x :: rM[x,y] == rM[y,x] && eM[x,y] == eM[y,x]
      {
        var j := 0;
        while j < i
          invariant 0 <= i <= Rows(m) && 0 <= j <= i
          invariant forall x,y | 0 <= x < i && 0 <= y < Cols(m) :: rM[x,y] == m[x][y] + eM[x,y]
          invariant forall x,y | 0 <= x < i && 0 <= y < x :: rM[x,y] == rM[y,x] && eM[x,y] == eM[y,x]
          invariant forall y | 0 <= y < j :: rM[i,y] == rM[y,i] && eM[i,y] == eM[y,i]
        {
          rM[i,j] := rM[j,i];
          eM[i,j] := eM[j,i];
          j := j + 1;
        }
        assert forall y | 0 <= y < i :: rM[i,y] == rM[y,i];

        while j < rM.Length1
          invariant 0 <= i <= Rows(m) && 0 <= j <= Cols(m)
          invariant forall x,y | 0 <= x < i && 0 <= y < Cols(m) :: rM[x,y] == m[x][y] + eM[x,y]
          invariant forall x,y | 0 <= x <= i && 0 <= y < x :: rM[x,y] == rM[y,x] && eM[x,y] == eM[y,x]
          invariant forall y | 0 <= y < j :: rM[i,y] == m[i][y] + eM[i,y]
        {
          if m[i][j] > 0.0 {
            rM[i,j] := RoundUp(m[i][j]);
          } else if m[i][j] < 0.0 {
            rM[i,j] := RoundDown(m[i][j]);
          } else {
            rM[i,j] := m[i][j];
          }
          eM[i,j] := rM[i,j] - m[i][j];
          j := j + 1;
        }
        i := i + 1;

      }
      // FIXME: these two lines cause Dafny to time out (set --verification-time to 30 to verify this)
      r := A2M(rM);
      e := A2M(eM);
      Assumption4(m,r,e);
    }
  }

  ghost function ExpandWF(a: seq<(real,real)>) : (b: bool)
  {
    forall i | 0 <= i < |a| :: a[i].0 >= 0.0 && a[i].1 >= 0.0
  }

  ghost opaque function Expand(a: seq<(real,real)>, v: real) : (r: real)
    requires ExpandWF(a)
    requires v >= 0.0
  {
    if a == [] then v
    else Expand(a[1..],Sqrt((v + a[0].1)*a[0].0))
  }

  lemma ExpandRecurse(a: seq<(real,real)>, v: real)
    requires ExpandWF(a)
    requires v >= 0.0
    requires a != []
    ensures Expand(a, v) == Expand(a[1..],Sqrt((v + a[0].1)*a[0].0))
  { reveal Expand(); }

  method ExpandImpl(a: seq<(real,real)>, v: real) returns (r: real)
    requires ExpandWF(a)
    requires v >= 0.0
    ensures r >= Expand(a,v)
  {
    reveal Expand();
    if a == [] {
      r := v;
      return;
    }
    else {
      var b := SqrtUpperBound((v + a[0].1)*a[0].0);
      r := ExpandImpl(a[1..],b);
      assert r >= Expand(a[1..],b);
      ExpandMono(a[1..],Sqrt((v + a[0].1)*a[0].0),b);
    }
  }

  lemma ExpandInit(G: Matrix, ex: seq<(real,real)>, G': Matrix)
    requires G' == G && ex == []
    ensures SpecNorm(G) <= Expand(ex, SpecNorm(G'))
  {
    reveal Expand();
  }

  lemma ExpandMono(a: seq<(real,real)>, v1: real, v2: real)
    requires 0.0 <= v1 <= v2
    requires ExpandWF(a)
    ensures Expand(a,v1) <= Expand(a,v2)
  {
    reveal Expand();
    if a == [] {
    } else {
      MonotonicSqrt(a[0].0*(v1 + a[0].1),a[0].0*(v2 + a[0].1));
      ExpandMono(a[1..],Sqrt(a[0].0*(v1 + a[0].1)),Sqrt(a[0].0*(v2 + a[0].1)));
    }
  }

  lemma ExtendExWellformed(r: real, e: real, ex: seq<(real,real)>)
    requires r >= 0.0 && e >= 0.0
    requires ExpandWF(ex)
    ensures ExpandWF([(r,e)]+ex)
  {
  }

  method Gram(m: Matrix) returns (r: Matrix)
    ensures Sqrt(SpecNorm(r)) >= SpecNorm(m)
    ensures Rows(r) == Cols(r)
    ensures MatrixSymmetric(r)
  {
    r := MTM(m);
    Assumption1(m);
  }

  function IsMatrixDiv(m: Matrix, r: Matrix, x: real) : (b: bool)
    requires x > 0.0
    requires Rows(r) == Rows(m) && Cols(r) == Cols(m)
  {
    forall i,j | 0 <= i < Rows(r) && 0 <= j < Cols(r) :: r[i][j] == m[i][j]/x
  }

  lemma MatrixDivLemma(m: Matrix, x: real, r: Matrix)
    requires x > 0.0
    requires r == MatrixDiv(m, x)
    ensures SpecNorm(m) <= SpecNorm(r)*x
    ensures Rows(r) == Rows(m) && Cols(r) == Cols(m)
    ensures IsMatrixDiv(m, r, x)
  {
    Assumption3(m, x);
  }


  lemma MatrixSymmetricDiv(m: Matrix, r: Matrix, x: real)
    requires Rows(r) == Rows(m) && Cols(r) == Cols(m)
    requires Rows(m) == Cols(m)
    requires MatrixSymmetric(m)
    requires x > 0.0
    requires IsMatrixDiv(m, r, x)
    ensures MatrixSymmetric(r)
  {
  }

  method Normalise(m: Matrix) returns (r: Matrix, scale_factor: real)
    requires Rows(m) == Cols(m)
    requires MatrixSymmetric(m)
    ensures Rows(r) == Rows(m) && Cols(r) == Cols(m)
    ensures scale_factor > 0.0
    ensures SpecNorm(m) <= SpecNorm(r)*scale_factor
    ensures MatrixSymmetric(r)
  {
    opaque ensures scale_factor > 0.0 {
      scale_factor := 1.0;
      if !MatrixIsZero(m) {
        SafeFrobeniusNorm(m);
        scale_factor := FrobeniusNormUpperBound(m);
      }
    }
    opaque ensures Rows(r) == Rows(m) && Cols(r) == Cols(m) && IsMatrixDiv(m,r,scale_factor)
    && SpecNorm(m) <= SpecNorm(r)*scale_factor {
      r := MatrixDivImpl(m, scale_factor);
      MatrixDivLemma(m, scale_factor, r);
    }
    MatrixSymmetricDiv(m,r,scale_factor);
  }

  lemma WTFLemma1(x: real, u: real, v: real, s: real, a: real, b: real)
    requires s >= 0.0
    requires x <= u + v
    requires a == x * s
    requires b == (u + v) * s
    ensures a <= b
  {
    MultIsMono(x,(u + v),s);
  }

  lemma ExpandMono2(x: real, ex: seq<(real,real)>, v1: real, v2: real)
    requires ExpandWF(ex)
    requires 0.0 <= v1
    requires x <= Expand(ex, v1)
    requires v1 <= v2
    ensures x <= Expand(ex, v2)
  {
    ExpandMono(ex,v1,v2);
  }

  lemma ExpandInductive(G: Matrix, old_ex: seq<(real,real)>, ex: seq<(real,real)>, b: real, G': Matrix, scale_factor: real, f: real)
    requires ExpandWF(old_ex)
    requires ExpandWF(ex)
    requires b >= 0.0
    requires SpecNorm(G) <= Expand(old_ex,Sqrt(b))
    requires b == (SpecNorm(G') + f)*scale_factor
    requires ex == [(scale_factor,f)]+old_ex
    requires scale_factor >= 0.0 && f >= 0.0
    ensures SpecNorm(G) <= Expand(ex, SpecNorm(G'))
  {

    ghost var v: real := SpecNorm(G');
    assert ex[1..] == old_ex;
    assert ex[0].0 == scale_factor;
    assert ex[0].1 == f;
    assert !(ex == []);
    ExpandRecurse(ex, SpecNorm(G'));
  }


  method GramIterationSimple(G: Matrix, GRAM_ITERATIONS: int) returns (s: real)
    requires GRAM_ITERATIONS >= 0
    ensures IsSpecNormUpperBound(s, G)
  {
    var i := 0;
    var G' := G;
    var ex: seq<(real,real)> := [];
    ExpandInit(G,ex,G');
    while i != GRAM_ITERATIONS
      invariant 0 <= i <= GRAM_ITERATIONS
      invariant ExpandWF(ex)
      invariant SpecNorm(G) <= Expand(ex, SpecNorm(G'))
    {
      ghost var oG' := G';
      assert SpecNorm(G) <= Expand(ex, SpecNorm(oG'));

      var G1: Matrix := Gram(G');
      ExpandMono2(SpecNorm(G),ex,SpecNorm(oG'),Sqrt(SpecNorm(G1)));
      var G2: Matrix, scale_factor: real := Normalise(G1);
      MonotonicSqrt(SpecNorm(G1),SpecNorm(G2)*scale_factor);
      ExpandMono2(SpecNorm(G),ex,Sqrt(SpecNorm(G1)),Sqrt(SpecNorm(G2)*scale_factor));

      var Trunc: Matrix;
      var E: Matrix;
      Trunc,E := Truncate(G2);

      Assumption2(E);
      var f := FrobeniusNormUpperBound(E);
      MultIsMono(SpecNorm(G2),SpecNorm(Trunc) + f,scale_factor); // SpecNorm(G2)*scale_factor <= (SpecNorm(Trunc) + f)*scale_factor

      ghost var x := SpecNorm(G2);
      ghost var u := SpecNorm(Trunc);
      ghost var a := x*scale_factor;
      ghost var b := (u + f)*scale_factor;
      WTFLemma1(x,u,f,scale_factor,a,b);

      MonotonicSqrt(a,b);
      ExpandMono2(SpecNorm(G),ex,Sqrt(a),Sqrt(b));
      G' := Trunc;
      i := i + 1;
      ExtendExWellformed(scale_factor,f,ex);
      var old_ex := ex;
      ex := [(scale_factor,f)]+old_ex;
      ExpandInductive(G,old_ex,ex,b,G',scale_factor,f);
    }
    Assumption2(G');
    ExpandMono(ex,SpecNorm(G'), FrobeniusNorm(G'));
    if DEBUG { print "{ \"debug_msg\": \"Gram iteration computing frobenius norm upper bound...\" },\n"; }
    s := FrobeniusNormUpperBound(G');
    ExpandMono(ex,FrobeniusNorm(G'),s);
    if DEBUG { print "{ \"debug_msg\": \"Gram iteration expanding...\" },\n"; }
    var ret := ExpandImpl(ex,s);
    calc {
      SpecNorm(G)
   <=
      Expand(ex,SpecNorm(G'))
   <=
      FrobeniusNorm(G')
   <=
      s
   <=
      ret;
    }
    SpecNormUpperBoundProperty(ret, G);
    s := ret;
    if DEBUG { print "{ \"debug_msg\": \"Gram iteration done\" },\n"; }
  }


  /* ================================= Lemmas ================================= */

  // ASSUMPTIONS
  lemma {:axiom} Assumption1(m: Matrix)
    ensures SpecNorm(m) <= Sqrt(SpecNorm(MM(Transpose(m), m)))

  lemma {:axiom} Assumption2(m: Matrix)
    ensures SpecNorm(m) <= FrobeniusNorm(m)

  lemma {:axiom} CauchySchwartz(v: Vector, u: Vector)
    requires |v| == |u|
    ensures Dot(v, u) * Dot(v, u) <= Dot(v, v) * Dot(u, u)

  /* We only need these for rescaling gram iteration */

  lemma {:axiom} Assumption3(m: Matrix, x: real)
    requires 0.0 < x
    ensures SpecNorm(m) <= SpecNorm(MatrixDiv(m, x)) * x

  lemma {:axiom} Assumption4(m: Matrix, n: Matrix, e: Matrix)
    requires Rows(m) == Rows(n) == Rows(e) && Cols(m) == Cols(n) == Cols(e) && Rows(n) == Cols(n)
    requires forall i,j | 0 <= i < Rows(m) && 0 <= j < Cols(m) :: m[i][j] == m[j][i] && e[i][j] == e[i][j]
    requires forall i,j | 0 <= i < Rows(n) && 0 <= j < Cols(n) :: n[i][j] == m[i][j] + e[i][j]
    ensures Abs(SpecNorm(n)-SpecNorm(m)) <= SpecNorm(e)


  ghost function MatrixDiv(m: Matrix, x: real): (r: Matrix)
    requires 0.0 < x
    ensures |r| == |m| && |r[0]| == |m[0]|
    ensures forall i,j | 0 <= i < |r| && 0 <= j < |r[0]| :: r[i][j] == m[i][j]/x
  {
    if |m| == 1 then [Apply(m[0], a => a/x)]
    else [Apply(m[0], a => a/x)] + MatrixDiv(m[1..], x)
  }

  method MatrixDivImpl(m: Matrix, x: real) returns (r: Matrix)
    requires 0.0 < x
    ensures r == MatrixDiv(m, x)
  {
    var i := |m| - 1;
    var b := ApplyImpl(m[i], a => a/x);
    r := [b];
    while i > 0
      invariant r == MatrixDiv(m[i..], x)
    {
      i := i - 1;
      b := ApplyImpl(m[i], a => a/x);
      r := [b] + r;
    }
  }

  // function VectorDiv(v: Vector, x: real): (r: Vector)
  //   requires 0.0 < x
  //   ensures |r| == |v|
  // {
  //   if |v| == 1 then [v[0] / x] else [v[0] / x] + VectorDiv(v[1..], x)
  // }

  lemma SpecNormUpperBoundProperty(s: real, m: Matrix)
    requires s >= SpecNorm(m)
    ensures s >= 0.0
    ensures IsSpecNormUpperBound(s, m)
  {
    PositiveL2();
  }

  /** 
    * If each element in v has a lower absolute value than its counterpart in u,
    * then ||v|| <= ||u||.
    */
  lemma SmallerL2Norm(v: Vector, u: Vector)
    requires |v| == |u|
    requires forall i: int :: 0 <= i < |v| ==> Abs(v[i]) <= Abs(u[i])
    ensures L2(v) <= L2(u)
  {
    reveal L2();
    SmallerApplySquare(v, u);
    MonotonicSum(Apply(v, Square), Apply(u, Square));
    MonotonicSqrt(Sum(Apply(v, Square)), Sum(Apply(u, Square)));
  }

  /** A larger number is also a spectral norm upper bound. */
  lemma SpecNormUpperBoundMono(b: real, s: real, m: Matrix)
    requires IsSpecNormUpperBound(b, m) && b <= s
    ensures IsSpecNormUpperBound(s, m)
  {
    PositiveL2();
    forall v: Vector | |v| == |m[0]|
      ensures L2(MV(m, v)) <= s * L2(v)
    {
      MultIsMono(b, s, L2(v));
    }
  }

  /* ============== Convolutional layers: Algorithm 3 (norm2_toep) ============== */

  /**
    * Algorithm 3 (norm2_toep) of Delattre et al. (2024), with truncation: an upper
    * bound on the spectral norm of the layer's banded doubly block Toeplitz
    * matrix, from N >= 1 Gram iterations on the layer's filter. As
    * GramIterationSimple does for matrices, each iteration takes the Gram filter
    * (line 6), rescales it by an upper bound on its Frobenius norm (lines 4-5,
    * done after the Gram step so that the step itself runs on rounded weights)
    * and rounds it to ROUNDING_PRECISION decimal places; without the rounding,
    * the numbers' digits double at every iteration. ex records the scale factors
    * and the rounding errors, and ExpandImpl undoes them after the bound of
    * line 7. The filter still grows from k x k to (2k-1) x (2k-1) at each
    * iteration, so N should be small.
    */
  method Norm2Toep(L: ConvLayer, N: nat) returns (s: real)
    requires ValidConv(L) && N >= 1
    ensures IsSpecNormUpperBound(s, ToeplitzMatrix(L))
  {
    // F, of size cO x L.cIn x k x k, is the filter after t iterations
    var F, cO, k := L.K, L.cOut, L.k;
    var ex: seq<(real,real)> := [];
    ConvCrop(L);
    ExpandNil(ConvNorm(FullConv(F, cO, L.cIn, k, L.n)));
    var t := 0;
    while t < N
      invariant 0 <= t <= N
      invariant IsFilter(F, cO, L.cIn, k) && cO > 0 && k > 0
      invariant ExpandWF(ex)
      invariant ConvNorm(L) <= Expand(ex, ConvNorm(FullConv(F, cO, L.cIn, k, L.n)))
    {
      var R, c, e := Norm2ToepStep(F, cO, L.cIn, k, L.n);
      ExpandStep(ConvNorm(L), ex, ConvNorm(FullConv(F, cO, L.cIn, k, L.n)),
                 ConvNorm(FullConv(R, L.cIn, L.cIn, 2 * k - 1, L.n)), c, e);
      ex := [(c, e)] + ex;
      F, cO, k := R, L.cIn, 2 * k - 1;
      t := t + 1;
    }
    // line 7, then undo the scale factors and add back the rounding errors
    var b := SchurBound(F, cO, L.cIn, k);
    ConvSchur(FullConv(F, cO, L.cIn, k, L.n));
    ExpandMono(ex, ConvNorm(FullConv(F, cO, L.cIn, k, L.n)), b);
    s := ExpandImpl(ex, b);
    ConvNormIsSpecNorm(L);
    SpecNormUpperBoundProperty(s, ToeplitzMatrix(L));
  }

  /**
    * One iteration of Norm2Toep on the filter F (cOut x cIn x k x k). R
    * (cIn x cIn x (2k-1) x (2k-1)) is the Gram filter of F divided by c and
    * rounded, and e bounds the rounding error's spectral norm, so that
    * ||T(F)|| <= sqrt((||T(R)|| + e) * c), where T(F) is the matrix of the
    * convolution with F with stride 1 and maximal non-trivial zero padding.
    */
  method Norm2ToepStep(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat, ghost n: nat)
    returns (R: seq<seq<seq<seq<real>>>>, c: real, e: real)
    requires IsFilter(F, cOut, cIn, k) && cOut > 0 && cIn > 0 && k > 0 && n > 0
    ensures IsFilter(R, cIn, cIn, 2 * k - 1)
    ensures c > 0.0 && e >= 0.0
    ensures ConvNorm(FullConv(F, cOut, cIn, k, n)) <=
            StepBound(ConvNorm(FullConv(R, cIn, cIn, 2 * k - 1, n)), e, c)
  {
    // line 6: Gram iteration
    var s := 2 * k - 1;
    var G := GramStep(F, cOut, cIn, k);
    // lines 4-5: rescale
    c := FilterScale(G, cIn, cIn, s);
    var H := FilterDiv(G, cIn, cIn, s, c);
    // truncation: D = H - R is the rounding error
    R := TruncateFilter(H, cIn, cIn, s, Pow10(ROUNDING_PRECISION));
    var D := FilterSub(H, R, cIn, cIn, s);
    e := SchurBound(D, cIn, cIn, s);

    ghost var nF := ConvNorm(FullConv(F, cOut, cIn, k, n));
    ghost var nG := ConvNorm(SameConv(G, cIn, s, n));
    ghost var nH := ConvNorm(SameConv(H, cIn, s, n));
    ghost var nR := ConvNorm(FullConv(R, cIn, cIn, s, n));
    GramFull(F, cOut, cIn, k, s, n);          // nF <= Sqrt(nG)
    ScaleSame(G, cIn, s, n, c);               // nG <= Times(nH, c)
    TriangleSame(H, R, cIn, s, n);            // nH <= ConvNorm(SameConv(R)) + ConvNorm(SameConv(D))
    SchurSame(D, cIn, s, n);                  // ConvNorm(SameConv(D)) <= e
    CropSame(R, cIn, s, n);                   // ConvNorm(SameConv(R)) <= nR
    Norm2ToepStepBound(nF, nG, nH, nR, e, c);
  }

  /**
    * An upper bound on the Frobenius norm of F, or 1 if F is zero: the scale
    * factor of lines 4-5. Any c > 0 is sound; this one keeps every weight of F/c
    * between -1 and 1.
    */
  method FilterScale(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat) returns (c: real)
    requires IsFilter(F, cOut, cIn, k)
    ensures c > 0.0
  {
    var sum := 0.0;
    for j := 0 to cOut
      invariant sum >= 0.0
    {
      for i := 0 to cIn
        invariant sum >= 0.0
      {
        for a := 0 to k
          invariant sum >= 0.0
        {
          for b := 0 to k
            invariant sum >= 0.0
          {
            sum := sum + F[j][i][a][b] * F[j][i][a][b];
          }
        }
      }
    }
    c := SqrtUpperBound(sum);
    if c <= 0.0 {
      c := 1.0;
    }
  }

  /**
    * F with every weight rounded away from zero to a multiple of 1/S, as
    * Truncate does for matrices with S = 10^ROUNDING_PRECISION. Equal weights
    * round to equal weights, so a Gram filter's symmetry is kept.
    */
  function TruncateFilter(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat, S: real):
    (R: seq<seq<seq<seq<real>>>>)
    requires IsFilter(F, cOut, cIn, k) && S > 0.0
    ensures IsFilter(R, cOut, cIn, k)
  {
    seq(cOut, j requires 0 <= j < cOut =>
      seq(cIn, i requires 0 <= i < cIn =>
        seq(k, a requires 0 <= a < k =>
          seq(k, b requires 0 <= b < k => RoundAway(F[j][i][a][b], S)))))
  }

  /**
    * x rounded away from zero to a multiple of 1/S: for S = 10^ROUNDING_PRECISION
    * the same as RoundUp and RoundDown, but with one Floor instead of a loop.
    */
  function RoundAway(x: real, S: real): real
    requires S > 0.0
  {
    var y := x * S;
    (if y >= 0.0 then -((-y).Floor) else y.Floor) as real / S
  }

  /** 10^n. */
  function Pow10(n: nat): (r: real)
    ensures r > 0.0
  {
    if n == 0 then 1.0 else 10.0 * Pow10(n - 1)
  }

  /** F with every weight divided by d. */
  function FilterDiv(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat, d: real):
    (H: seq<seq<seq<seq<real>>>>)
    requires IsFilter(F, cOut, cIn, k) && d > 0.0
    ensures IsFilter(H, cOut, cIn, k)
    ensures forall j, i, a, b | 0 <= j < cOut && 0 <= i < cIn && 0 <= a < k && 0 <= b < k ::
              H[j][i][a][b] == F[j][i][a][b] / d
  {
    seq(cOut, j requires 0 <= j < cOut =>
      seq(cIn, i requires 0 <= i < cIn =>
        seq(k, a requires 0 <= a < k =>
          seq(k, b requires 0 <= b < k => F[j][i][a][b] / d))))
  }

  /** F - R, weight by weight. */
  function FilterSub(F: seq<seq<seq<seq<real>>>>, R: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat):
    (D: seq<seq<seq<seq<real>>>>)
    requires IsFilter(F, cOut, cIn, k) && IsFilter(R, cOut, cIn, k)
    ensures IsFilter(D, cOut, cIn, k)
    ensures forall j, i, a, b | 0 <= j < cOut && 0 <= i < cIn && 0 <= a < k && 0 <= b < k ::
              D[j][i][a][b] == F[j][i][a][b] - R[j][i][a][b]
  {
    seq(cOut, j requires 0 <= j < cOut =>
      seq(cIn, i requires 0 <= i < cIn =>
        seq(k, a requires 0 <= a < k =>
          seq(k, b requires 0 <= b < k => F[j][i][a][b] - R[j][i][a][b]))))
  }

  /**
    * The spectral norm of the layer's matrix. Opaque, so that the matrix is
    * unfolded only in the lemmas that need it.
    */
  ghost opaque function ConvNorm(L: ConvLayer): (r: real)
    requires ValidConv(L)
    ensures r >= 0.0
  {
    SpecNorm(ToeplitzMatrix(L))
  }

  /**
    * a * b, and the bound sqrt((w + e) * c) of one step of Expand. Opaque, so
    * that the verifier matches them by their arguments rather than by
    * non-linear arithmetic.
    */
  ghost opaque function Times(a: real, b: real): real
  {
    a * b
  }

  ghost opaque function StepBound(w: real, e: real, c: real): real
    requires w >= 0.0 && e >= 0.0 && c >= 0.0
  {
    Sqrt((w + e) * c)
  }

  /**
    * The convolution with filter F (cOut x cIn x k x k) on an n x n input, with
    * stride 1 and maximal non-trivial zero padding: k-1 zeros on each side, so
    * the input is padded to (n+2k-2) x (n+2k-2) (Prach & Lampert 2022, App. A.1).
    */
  ghost function FullConv(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat, n: nat): (L: ConvLayer)
    requires IsFilter(F, cOut, cIn, k) && cOut > 0 && cIn > 0 && k > 0 && n > 0
    ensures ValidConv(L)
  {
    ConvLayer(F, cIn, n, cOut, k, 1, k - 1)
  }

  /**
    * The convolution with filter F (c x c x k x k, k odd) on an n x n input, with
    * stride 1 and (k-1)/2 zeros on each side, so the output is n x n too.
    */
  ghost function SameConv(F: seq<seq<seq<seq<real>>>>, c: nat, k: nat, n: nat): (L: ConvLayer)
    requires IsFilter(F, c, c, k) && c > 0 && k % 2 == 1 && n > 0
    ensures ValidConv(L)
  {
    ConvLayer(F, c, n, c, k, 1, (k - 1) / 2)
  }

  /** The layer L with filter R in place of L.K. */
  ghost function WithFilter(L: ConvLayer, R: seq<seq<seq<seq<real>>>>): (L': ConvLayer)
    requires ValidConv(L) && IsFilter(R, L.cOut, L.cIn, L.k)
    ensures ValidConv(L')
  {
    L.(K := R)
  }

  /* -------------------------------- Lemmas -------------------------------- */

  lemma ExpandNil(v: real)
    requires v >= 0.0
    ensures Expand([], v) == v
  {
    reveal Expand();
  }

  /** Adds an iteration's scale factor c and rounding error e to ex. */
  lemma ExpandStep(x: real, ex: seq<(real,real)>, v: real, w: real, c: real, e: real)
    requires ExpandWF(ex) && v >= 0.0 && w >= 0.0 && c >= 0.0 && e >= 0.0
    requires x <= Expand(ex, v)
    requires v <= StepBound(w, e, c)
    ensures ExpandWF([(c, e)] + ex)
    ensures x <= Expand([(c, e)] + ex, w)
  {
    reveal StepBound();
    var ex' := [(c, e)] + ex;
    assert ex'[1..] == ex;
    ExpandMono(ex, v, Sqrt((w + e) * c));
    ExpandRecurse(ex', w);
  }

  /** The arithmetic of Norm2ToepStep. */
  lemma Norm2ToepStepBound(nF: real, nG: real, nH: real, nR: real, e: real, c: real)
    requires nG >= 0.0 && nH >= 0.0 && nR >= 0.0 && e >= 0.0 && c >= 0.0
    requires nF <= Sqrt(nG) && nG <= Times(nH, c) && nH <= nR + e
    ensures nF <= StepBound(nR, e, c)
  {
    TimesMono(nH, nR + e, c);
    MonotonicSqrt(nG, Times(nR + e, c));
    StepBoundTimes(nR, e, c);
  }

  lemma TimesMono(a: real, b: real, c: real)
    requires a <= b && c >= 0.0
    ensures Times(a, c) <= Times(b, c)
  {
    reveal Times();
    MultIsMono(a, b, c);
  }

  lemma StepBoundTimes(w: real, e: real, c: real)
    requires w >= 0.0 && e >= 0.0 && c >= 0.0
    ensures Times(w + e, c) >= 0.0
    ensures StepBound(w, e, c) == Sqrt(Times(w + e, c))
  {
    reveal StepBound(), Times();
  }

  lemma ConvNormIsSpecNorm(L: ConvLayer)
    requires ValidConv(L)
    ensures ConvNorm(L) == SpecNorm(ToeplitzMatrix(L))
  {
    reveal ConvNorm();
  }

  /** Lines 4-5: ConvScale for the Gram filter G. */
  lemma ScaleSame(G: seq<seq<seq<seq<real>>>>, c: nat, s: nat, n: nat, d: real)
    requires IsFilter(G, c, c, s) && c > 0 && s % 2 == 1 && n > 0 && d > 0.0
    ensures ConvNorm(SameConv(G, c, s, n)) <=
            Times(ConvNorm(SameConv(FilterDiv(G, c, c, s, d), c, s, n)), d)
  {
    ConvScale(SameConv(G, c, s, n), d);
  }

  /** Line 6: ConvGram, with the Gram filter's size s = 2k-1. */
  lemma GramFull(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat, s: nat, n: nat)
    requires IsFilter(F, cOut, cIn, k) && cOut > 0 && cIn > 0 && k > 0 && n > 0 && s == 2 * k - 1
    ensures ConvNorm(FullConv(F, cOut, cIn, k, n)) <=
            Sqrt(ConvNorm(SameConv(GramStep(F, cOut, cIn, k), cIn, s, n)))
  {
    ConvGram(F, cOut, cIn, k, n);
  }

  /** Truncation: ConvTriangle for G = R + (G - R). */
  lemma TriangleSame(G: seq<seq<seq<seq<real>>>>, R: seq<seq<seq<seq<real>>>>, c: nat, s: nat, n: nat)
    requires IsFilter(G, c, c, s) && IsFilter(R, c, c, s) && c > 0 && s % 2 == 1 && n > 0
    ensures ConvNorm(SameConv(G, c, s, n)) <=
            ConvNorm(SameConv(R, c, s, n)) + ConvNorm(SameConv(FilterSub(G, R, c, c, s), c, s, n))
  {
    ConvTriangle(SameConv(G, c, s, n), R);
  }

  lemma SchurSame(D: seq<seq<seq<seq<real>>>>, c: nat, s: nat, n: nat)
    requires IsFilter(D, c, c, s) && c > 0 && s % 2 == 1 && n > 0
    ensures ConvNorm(SameConv(D, c, s, n)) <= SchurBound(D, c, c, s)
  {
    ConvSchur(SameConv(D, c, s, n));
  }

  lemma CropSame(R: seq<seq<seq<seq<real>>>>, c: nat, s: nat, n: nat)
    requires IsFilter(R, c, c, s) && c > 0 && s % 2 == 1 && n > 0
    ensures ConvNorm(SameConv(R, c, s, n)) <= ConvNorm(FullConv(R, c, c, s, n))
  {
    ConvCrop(SameConv(R, c, s, n));
  }

  /** Dividing the filter by d > 0 divides the layer's matrix by d. */
  lemma ConvScale(L: ConvLayer, d: real)
    requires ValidConv(L) && d > 0.0
    ensures ConvNorm(L) <= Times(ConvNorm(WithFilter(L, FilterDiv(L.K, L.cOut, L.cIn, L.k, d))), d)
  {
    var L' := WithFilter(L, FilterDiv(L.K, L.cOut, L.cIn, L.k, d));
    ToeplitzDiv(L, d);
    SpecNormDiv(ToeplitzMatrix(L), ToeplitzMatrix(L'), d);
    reveal ConvNorm();
  }

  /** Assumption3, as MatrixDivLemma states it. */
  lemma SpecNormDiv(m: Matrix, m': Matrix, d: real)
    requires d > 0.0 && m' == MatrixDiv(m, d)
    ensures SpecNorm(m) <= Times(SpecNorm(m'), d)
  {
    MatrixDivLemma(m, d, m');
    reveal Times();
  }

  lemma ToeplitzDiv(L: ConvLayer, d: real)
    requires ValidConv(L) && d > 0.0
    ensures ToeplitzMatrix(WithFilter(L, FilterDiv(L.K, L.cOut, L.cIn, L.k, d))) ==
            MatrixDiv(ToeplitzMatrix(L), d)
  {
    var L' := WithFilter(L, FilterDiv(L.K, L.cOut, L.cIn, L.k, d));
    var T, T', M := ToeplitzMatrix(L), ToeplitzMatrix(L'), MatrixDiv(ToeplitzMatrix(L), d);
    var m, nn := OutSize(L), L.n * L.n;
    assert OutSize(L') == m;
    forall r | 0 <= r < |T'|
      ensures T'[r] == M[r]
    {
      forall col | 0 <= col < |T'[r]|
        ensures T'[r][col] == M[r][col]
      {
        DivBound(r, L.cOut, m * m);
        DivBound(col, L.cIn, nn);
        assert T'[r][col] == ToeplitzEntry(L', r / (m * m), (r / m) % m, r % m,
                                           col / nn, (col / L.n) % L.n, col % L.n);
        assert T[r][col] == ToeplitzEntry(L, r / (m * m), (r / m) % m, r % m,
                                          col / nn, (col / L.n) % L.n, col % L.n);
      }
    }
  }

  /**
    * The layer's matrix is linear in its filter, so the triangle inequality for
    * the spectral norm holds for its filter: ||T(K)|| <= ||T(R)|| + ||T(K - R)||.
    */
  lemma ConvTriangle(L: ConvLayer, R: seq<seq<seq<seq<real>>>>)
    requires ValidConv(L) && IsFilter(R, L.cOut, L.cIn, L.k)
    ensures ConvNorm(L) <=
            ConvNorm(WithFilter(L, R)) + ConvNorm(WithFilter(L, FilterSub(L.K, R, L.cOut, L.cIn, L.k)))
  {
    reveal ConvNorm();
    var LR := WithFilter(L, R);
    var LD := WithFilter(L, FilterSub(L.K, R, L.cOut, L.cIn, L.k));
    var T, TR, TD := ToeplitzMatrix(L), ToeplitzMatrix(LR), ToeplitzMatrix(LD);
    var m, nn := OutSize(L), L.n * L.n;
    assert OutSize(LR) == m && OutSize(LD) == m;
    forall r, col | 0 <= r < Rows(T) && 0 <= col < Cols(T)
      ensures T[r][col] == TR[r][col] + TD[r][col]
    {
      DivBound(r, L.cOut, m * m);
      DivBound(col, L.cIn, nn);
      assert T[r][col] == ToeplitzEntry(L, r / (m * m), (r / m) % m, r % m,
                                        col / nn, (col / L.n) % L.n, col % L.n);
      assert TR[r][col] == ToeplitzEntry(LR, r / (m * m), (r / m) % m, r % m,
                                         col / nn, (col / L.n) % L.n, col % L.n);
      assert TD[r][col] == ToeplitzEntry(LD, r / (m * m), (r / m) % m, r % m,
                                         col / nn, (col / L.n) % L.n, col % L.n);
    }
    SpecNormTriangle(T, TR, TD);
  }

  /** One Gram iteration on the filter: Assumption1 and ConvGramIdentity. */
  lemma ConvGram(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat, n: nat)
    requires IsFilter(F, cOut, cIn, k) && cOut > 0 && cIn > 0 && k > 0 && n > 0
    ensures ConvNorm(FullConv(F, cOut, cIn, k, n)) <=
            Sqrt(ConvNorm(SameConv(GramStep(F, cOut, cIn, k), cIn, 2 * k - 1, n)))
  {
    reveal ConvNorm();
    ConvGramIdentity(F, cOut, cIn, k, n);
    Assumption1(ToeplitzMatrix(FullConv(F, cOut, cIn, k, n)));
  }

  /* ------------------------------ Assumptions ------------------------------ */

  /**
    * ASSUMPTION (Prach & Lampert 2022, Eq. 17): with stride 1 and maximal
    * non-trivial zero padding, T^T T, where T is the convolution's matrix, is the
    * matrix of the convolution with the Gram filter GramStep(F) of size 2k-1 and
    * k-1 zeros of padding on each side. This is the step of Theorem 4 of Delattre
    * et al. (2024).
    */
  lemma {:axiom} ConvGramIdentity(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat, n: nat)
    requires IsFilter(F, cOut, cIn, k) && cOut > 0 && cIn > 0 && k > 0 && n > 0
    ensures MM(Transpose(ToeplitzMatrix(FullConv(F, cOut, cIn, k, n))),
               ToeplitzMatrix(FullConv(F, cOut, cIn, k, n))) ==
            ToeplitzMatrix(SameConv(GramStep(F, cOut, cIn, k), cIn, 2 * k - 1, n))

  /**
    * ASSUMPTION (triangle inequality): the spectral norm of a sum of two matrices
    * is at most the sum of their spectral norms. Assumption4 says the same, but
    * only for a symmetric matrix.
    */
  lemma {:axiom} SpecNormTriangle(m: Matrix, n: Matrix, e: Matrix)
    requires Rows(m) == Rows(n) == Rows(e) && Cols(m) == Cols(n) == Cols(e)
    requires forall i, j | 0 <= i < Rows(m) && 0 <= j < Cols(m) :: m[i][j] == n[i][j] + e[i][j]
    ensures SpecNorm(m) <= SpecNorm(n) + SpecNorm(e)

  /**
    * ASSUMPTION (Prach & Lampert 2022, App. A.1): each row of the layer's matrix
    * is zero or a row of the matrix of the same filter with stride 1 and maximal
    * non-trivial zero padding, and no row of that matrix is used twice. By the
    * interlacing inequalities for singular values of submatrices (Thompson 1972),
    * any other zero padding or stride therefore cannot increase the spectral norm.
    */
  lemma {:axiom} ConvCrop(L: ConvLayer)
    requires ValidConv(L)
    ensures ConvNorm(L) <= ConvNorm(FullConv(L.K, L.cOut, L.cIn, L.k, L.n))

  /**
    * ASSUMPTION (Schur's test): a matrix's spectral norm is at most the square
    * root of its largest absolute column sum times its largest absolute row sum.
    * Each column of the layer's matrix holds each weight of its input channel at
    * most once, and each row each weight of its output channel, so
    * MaxAbsColumnSum and MaxAbsRowSum bound these sums, and SchurBound, the
    * larger of the two, bounds the spectral norm, for any filter, zero padding
    * and stride.
    */
  lemma {:axiom} ConvSchur(L: ConvLayer)
    requires ValidConv(L)
    ensures ConvNorm(L) <= SchurBound(L.K, L.cOut, L.cIn, L.k)

  /* ------------------------------- Filters -------------------------------- */

  /**
    * F[j][i][a][b], with j < cOut, i < cIn and a, b < k, is the k x k kernel from
    * input channel i to output channel j, as Delattre et al. (2024) index filters
    * (a ConvLayer's K[o][c][a][b]).
    */
  predicate IsFilter(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat) {
    |F| == cOut &&
    forall j | 0 <= j < cOut :: |F[j]| == cIn &&
                                forall i | 0 <= i < cIn :: |F[j][i]| == k &&
                                                           forall a | 0 <= a < k :: |F[j][i][a]| == k
  }

  /**
    * One Gram iteration on the filter (Theorem 4): G[i1][i2] is the sum over j of
    * the cross-correlation of F[j][i1] and F[j][i2], at all (2k-1) x (2k-1)
    * offsets: G[i1][i2][u][v] = sum of F[j][i1][a][b] * F[j][i2][a+u-k+1][b+v-k+1]
    * over j, a and b. With maximal non-trivial zero padding (k-1 zeros on each
    * side) and stride 1, it is the filter of T^T T, where T is the matrix of the
    * convolution with F (Prach & Lampert 2022, Eq. 17).
    */
  function GramStep(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat):
    (G: seq<seq<seq<seq<real>>>>)
    requires IsFilter(F, cOut, cIn, k) && k > 0
    ensures IsFilter(G, cIn, cIn, 2 * k - 1)
  {
    seq(cIn, i1 requires 0 <= i1 < cIn =>
      seq(cIn, i2 requires 0 <= i2 < cIn =>
        seq(2 * k - 1, u requires 0 <= u < 2 * k - 1 =>
          seq(2 * k - 1, v requires 0 <= v < 2 * k - 1 =>
            CorrelationSum(F, cOut, cIn, k, i1, i2, u - k + 1, v - k + 1, cOut)))))
  }

  /** The sum over j < J, a < k and b < k of F[j][i1][a][b] * F[j][i2][a+du][b+dv]. */
  function CorrelationSum(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat,
                          i1: nat, i2: nat, du: int, dv: int, J: nat): real
    requires IsFilter(F, cOut, cIn, k) && i1 < cIn && i2 < cIn && J <= cOut
  {
    if J == 0 then 0.0
    else CorrelationSum(F, cOut, cIn, k, i1, i2, du, dv, J - 1) +
         CorrelationRows(F, cOut, cIn, k, J - 1, i1, i2, du, dv, k)
  }

  /** The sum over a < A and b < k, for one j. */
  function CorrelationRows(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat,
                           j: nat, i1: nat, i2: nat, du: int, dv: int, A: nat): real
    requires IsFilter(F, cOut, cIn, k) && j < cOut && i1 < cIn && i2 < cIn && A <= k
  {
    if A == 0 then 0.0
    else CorrelationRows(F, cOut, cIn, k, j, i1, i2, du, dv, A - 1) +
         CorrelationRow(F, cOut, cIn, k, j, i1, i2, A - 1, du, dv, k)
  }

  /** The sum over b < B, for one j and a. */
  function CorrelationRow(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat,
                          j: nat, i1: nat, i2: nat, a: nat, du: int, dv: int, B: nat): real
    requires IsFilter(F, cOut, cIn, k) && j < cOut && i1 < cIn && i2 < cIn && a < k && B <= k
  {
    if B == 0 then 0.0
    else
      var a2, b2 := a + du, B - 1 + dv;
      CorrelationRow(F, cOut, cIn, k, j, i1, i2, a, du, dv, B - 1) +
      (if 0 <= a2 < k && 0 <= b2 < k then F[j][i1][a][B - 1] * F[j][i2][a2][b2] else 0.0)
  }

  /**
    * Line 7 of Algorithm 3, before the 2^N-th root, made to hold for any filter:
    * the larger of MaxAbsColumnSum and MaxAbsRowSum. The two are equal for a
    * Gram filter, rounded or not.
    */
  function SchurBound(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat): (r: real)
    requires IsFilter(F, cOut, cIn, k)
    ensures r >= 0.0
  {
    var col := MaxAbsColumnSum(F, cOut, cIn, k);
    var row := MaxAbsRowSum(F, cOut, cIn, k);
    if col > row then col else row
  }

  /**
    * Line 7 of Algorithm 3, before the 2^N-th root: the largest, over input
    * channels i, of the sum of |F[j][i][a][b]| over j, a and b.
    */
  function MaxAbsColumnSum(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat): (r: real)
    requires IsFilter(F, cOut, cIn, k)
    ensures r >= 0.0
  {
    MaxAbsColumnSumUpTo(F, cOut, cIn, k, cIn)
  }

  function MaxAbsColumnSumUpTo(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat, I: nat): (r: real)
    requires IsFilter(F, cOut, cIn, k) && I <= cIn
    ensures r >= 0.0
  {
    if I == 0 then 0.0
    else
      var m := MaxAbsColumnSumUpTo(F, cOut, cIn, k, I - 1);
      var c := AbsColumnSum(F, cOut, cIn, k, I - 1, cOut);
      if c > m then c else m
  }

  /** The sum of |F[j][i][a][b]| over j < J, a < k and b < k, for one i. */
  function AbsColumnSum(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat, i: nat, J: nat): (r: real)
    requires IsFilter(F, cOut, cIn, k) && i < cIn && J <= cOut
    ensures r >= 0.0
  {
    if J == 0 then 0.0
    else AbsColumnSum(F, cOut, cIn, k, i, J - 1) + AbsKernelSum(F[J - 1][i], k, k)
  }

  /** The largest, over output channels j, of the sum of |F[j][i][a][b]| over i, a and b. */
  function MaxAbsRowSum(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat): (r: real)
    requires IsFilter(F, cOut, cIn, k)
    ensures r >= 0.0
  {
    MaxAbsRowSumUpTo(F, cOut, cIn, k, cOut)
  }

  function MaxAbsRowSumUpTo(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat, J: nat): (r: real)
    requires IsFilter(F, cOut, cIn, k) && J <= cOut
    ensures r >= 0.0
  {
    if J == 0 then 0.0
    else
      var m := MaxAbsRowSumUpTo(F, cOut, cIn, k, J - 1);
      var c := AbsOutputSum(F, cOut, cIn, k, J - 1, cIn);
      if c > m then c else m
  }

  /** The sum of |F[j][i][a][b]| over i < I, a < k and b < k, for one j. */
  function AbsOutputSum(F: seq<seq<seq<seq<real>>>>, cOut: nat, cIn: nat, k: nat, j: nat, I: nat): (r: real)
    requires IsFilter(F, cOut, cIn, k) && j < cOut && I <= cIn
    ensures r >= 0.0
  {
    if I == 0 then 0.0
    else AbsOutputSum(F, cOut, cIn, k, j, I - 1) + AbsKernelSum(F[j][I - 1], k, k)
  }

  /** The sum of |w[a][b]| over a < A and b < k. */
  function AbsKernelSum(w: seq<seq<real>>, k: nat, A: nat): (r: real)
    requires |w| == k && A <= k && forall a | 0 <= a < k :: |w[a]| == k
    ensures r >= 0.0
  {
    if A == 0 then 0.0 else AbsKernelSum(w, k, A - 1) + AbsRowSum(w[A - 1], k)
  }

  /** The sum of |x[b]| over b < B. */
  function AbsRowSum(x: seq<real>, B: nat): (r: real)
    requires B <= |x|
    ensures r >= 0.0
  {
    if B == 0 then 0.0 else AbsRowSum(x, B - 1) + (if x[B - 1] >= 0.0 then x[B - 1] else -x[B - 1])
  }

}
