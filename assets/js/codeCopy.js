// assets/js/post.js
var codeBlocks = document.querySelectorAll('div.highlight');

codeBlocks.forEach(function (codeBlock) {
  var copyButton = document.createElement('button');
  copyButton.className = 'copy-button';
  copyButton.type = 'button';
  copyButton.ariaLabel = 'Copy code to clipboard';
  copyButton.innerText = "copy";

  codeBlock.prepend(copyButton);
  copyButton.addEventListener('click', function () {
    var code = codeBlock.querySelector('code').innerText.trim();
    window.navigator.clipboard.writeText(code);

    copyButton.innerText = 'Copied';
    var fourSeconds = 1000;

    setTimeout(function () {
      copyButton.innerText = 'Copy';
    }, fourSeconds);
  });
});

