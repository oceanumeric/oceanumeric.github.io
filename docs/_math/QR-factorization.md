---
title: QR Factorization
subtitle: A QR factorization is a factorization of a matrix A into a product A = QR of an orthogonal matrix Q and an upper triangular matrix R. This kind of decomposition is useful in solving linear least squares problems and in the eigendecomposition of a matrix, which shows the structure of the matrix in terms of its eigenvalues and eigenvectors.
layout: math_page_template
date: 2023-04-02
keywords: QR factorization, orthogonal matrix, upper triangular matrix, linear least squares, eigendecomposition, eigenvalues, eigenvectors
published: true
tags: algorithm data-science machine-learning numerical-linear-algebra python
---

QR factorization is a factorization of a matrix A into a product A = QR of an orthogonal matrix $Q$ and an upper triangular matrix R. Since it reveals very useful information about the matrix A, it is a very important factorization. It is used in solving linear least squares problems and in the eigendecomposition of a matrix, which shows the structure of the matrix in terms of its eigenvalues and eigenvectors.


## Definition


{% katexmm %}

Recall that a matrix $Q$ is an _orthogonal matrix_ if $Q^T Q = I$. If $Q$ 
is orthogonal, then $||Qx||_2 = ||x||_2$ for all $x$; that is, $Q$ 
does not change the length of vectors. The operation that don't change 
the length of vectors are rotations and reflections, so an orthogonal 
matrix can then be thought of as a map that combines rotations and reflections. 

__QR factorization.__ Let $A$ be an $m \times n$ __real__ matrix where
$m \geq n$. Then there is an orthogonal matrix $Q$ and an upper-triangular
matrix $R$ so that $A = QR$. This is called the __QR factorization__. 

When $A$ is a __complex__ matrix, we can still write $A = QR$, where 
$Q$ is unitary instead of orthogonal. 

For the rest of this chapter, we will assume that $A$ is real, but it's 
important to know that $QR$ decomposition works for complex matrices 
as well. 












{% endkatexmm %}