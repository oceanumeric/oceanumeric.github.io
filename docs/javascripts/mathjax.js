window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => { 
  MathJax.typesetPromise()
}); 

if (window.MathJax) {
  MathJax.Hub.Queue(
    ["resetEquationNumbers",MathJax.InputJax.TeX],
    ["Typeset",MathJax.Hub]
  );
};
